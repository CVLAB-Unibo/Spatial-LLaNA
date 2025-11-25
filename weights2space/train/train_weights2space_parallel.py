'''

Adapted from https://github.com/CVLAB-Unibo/nf2vec

'''


import os
import sys

import torch.utils
from dataset import ShapeNeRFObjaNeRFDataset
from nerf.instant_ngp import NGPradianceField
from nerf.loader_gt import ShapeNeRFObjaNeRFLoaderGT

import math
from tqdm import tqdm
from nerfacc import OccupancyGrid
from utils import get_latest_checkpoints_path
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, DistributedSampler
from models.weights2space import Weights2Space
from pathlib import Path
from typing import Any, Dict
import cfg.original as config
from utils import Rays, render_image_shapenerf_objanerf_triplane, render_image_GT_shapenerf_objanerf

# wandb stuff
import wandb
import datetime
import numpy as np

# Pytorch Parallelism
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI


def setup(rank, world_size):
    print('Setting up...')
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=7200))
    print('Setup completed')

def cleanup():
    dist.destroy_process_group()


class ShapeNeRFObjaNeRFTrainerParallel:
    def __init__(self, rank, world_size, device='cuda:0', run_name='', run_id=None) -> None:
        self.rank = rank
        self.world_size = world_size
        self.process_group = dist.new_group(ranks=list(range(self.world_size)), timeout=datetime.timedelta(seconds=7200))
        self.device = device
        self.run_name = run_name
        self.run_id = run_id

        train_dset = ShapeNeRFObjaNeRFDataset('data/train.json', device='cpu')        
        self.train_sampler = DistributedSampler(train_dset, num_replicas=world_size, rank=rank)
        self.train_loader = DataLoader(
            train_dset,
            batch_size=config.BATCH_SIZE,
            shuffle=False, 
            num_workers=config.BATCH_SIZE,
            sampler=self.train_sampler
        )

        val_dset = ShapeNeRFObjaNeRFDataset('data/validation.json', device='cpu')   
        self.val_sampler = DistributedSampler(val_dset, num_replicas=world_size, rank=rank)
        self.val_loader = DataLoader(
            val_dset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.BATCH_SIZE,
            sampler=self.val_sampler
        )
        
        weights2space = Weights2Space(device, config)

        if config.SYNC_BATCHNORM:  # discussed, but not used in "Training ImageNet in 1 Hour"
            print('Converting to SyncBatchNorm...')
            weights2space = torch.nn.SyncBatchNorm.convert_sync_batchnorm(weights2space, self.process_group)
        
        self.weights2space = weights2space.to(self.device)
        self.weights2space = DDP(weights2space, device_ids=[self.device.index])

        occupancy_grid_shapenerf = OccupancyGrid(
            roi_aabb=config.GRID_AABB,
            resolution=config.GRID_RESOLUTION_SHAPENERF,
            contraction_type=config.GRID_CONTRACTION_TYPE,
        )
        self.occupancy_grid_shapenerf = occupancy_grid_shapenerf.to(self.device)
        self.occupancy_grid_shapenerf.eval()
        
        occupancy_grid_objanerf = OccupancyGrid(
            roi_aabb=config.GRID_AABB,
            resolution=config.GRID_RESOLUTION_OBJANERF,
            contraction_type=config.GRID_CONTRACTION_TYPE,
        )
        self.occupancy_grid_objanerf = occupancy_grid_objanerf.to(self.device)
        self.occupancy_grid_objanerf.eval()

        self.ngp_mlp = NGPradianceField(**config.INSTANT_NGP_MLP_CONF).to(device)
        self.ngp_mlp.eval()

        self.scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=self.device)
        self.render_step_size = (
            (self.scene_aabb[3:] - self.scene_aabb[:3]).max()
            * math.sqrt(3)
            / config.GRID_CONFIG_N_SAMPLES
        ).item()

        lr = config.LR * self.world_size  # linear scaling: "Training ImageNet in 1 Hour" # * 3 if we are using batch_size=48
        wd = config.WD        
        
        self.optimizer = AdamW(weights2space.parameters(), lr, weight_decay=wd)
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)
        self.scheduler = self.linear_warmup_scheduler(self.optimizer, warmup_epochs=28, start_lr=config.LR, target_lr=lr) # linear warmup: "Training ImageNet in 1 Hour"

        self.epoch = 0
        self.global_step = 0
        self.best_psnr = float("-inf")

        ckpts_root = 'weights2space/ckpts'
        os.makedirs(os.path.join(ckpts_root, self.run_name), exist_ok=True)

        self.ckpts_path = Path(os.path.join(ckpts_root, self.run_name, 'train', 'ckpts'))
        self.all_ckpts_path = Path(os.path.join(ckpts_root, self.run_name, 'train', 'all_ckpts'))
        
        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(parents=True, exist_ok=True)
        self.all_ckpts_path.mkdir(parents=True, exist_ok=True)

    def logfn(self, values: Dict[str, Any]) -> None:
        if self.rank == 0:  # Only log from the main process
            wandb.log(values, step=self.global_step, commit=False)

    def train(self):
        torch.backends.cudnn.enabled = False    # added by me. See https://discuss.pytorch.org/t/why-would-syncbatchnorm-give-different-results-from-batchnorm/146135
        
        if self.rank == 0:
            self.config_wandb(run_name=self.run_name, run_id=self.run_id)

        num_epochs = config.NUM_EPOCHS
        start_epoch = self.epoch
        
        for epoch in tqdm(range(start_epoch, num_epochs), desc='Epochs'):
            
            self.epoch = epoch
            self.weights2space.train()            
            #print(f'Process {self.rank} starting epoch {epoch}...')          
            self.train_loader.sampler.set_epoch(epoch)    # To make sure data is shuffled differently each epoch
            
            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                train_nerf, mlp_weights, mlp_matrix, data_dir, background_indices, _ = batch
                rays = train_nerf['rays']
                color_bkgds = train_nerf['color_bkgd']
                color_bkgds = color_bkgds[0][None].expand(len(mlp_matrix), -1)

                rays = rays._replace(origins=rays.origins.to(self.device), viewdirs=rays.viewdirs.to(self.device))
                color_bkgds = color_bkgds.to(self.device)
                mlp_matrix = mlp_matrix.to(self.device)

                #with autocast():
                pixels, alpha, filtered_rays, broken_batch_idxs = render_image_GT_shapenerf_objanerf(
                        radiance_field=self.ngp_mlp, 
                        occupancy_grid_shapenerf=self.occupancy_grid_shapenerf,
                        occupancy_grid_objanerf=self.occupancy_grid_objanerf,
                        rays=rays, 
                        scene_aabb=self.scene_aabb, 
                        render_step_size=self.render_step_size,
                        color_bkgds=color_bkgds,
                        ngp_mlp_weights=mlp_weights,
                        device=self.device,
                        data_dirs=data_dir)

                pixels = pixels * alpha + color_bkgds.unsqueeze(1) * (1.0 - alpha)

                triplane = self.weights2space(mlp_matrix)

                rgb, _, _, _,  bg_rgb_pred, bg_rgb_label = render_image_shapenerf_objanerf_triplane(
                    self.weights2space.module.decoder, 
                    triplane,
                    self.occupancy_grid_shapenerf,
                    self.occupancy_grid_objanerf,
                    filtered_rays,
                    self.scene_aabb,
                    render_step_size=self.render_step_size,
                    render_bkgd=color_bkgds,
                    background_indices=background_indices,
                    max_foreground_coordinates=config.MAX_FOREGROUND_COORDINATES,
                    max_background_coordinates=config.MAX_BACKGROUND_COORDINATES,
                    device=self.device,
                    data_dirs=data_dir
                )                        
                fg_loss = F.smooth_l1_loss(rgb, pixels) * config.FG_WEIGHT
                bg_loss = F.smooth_l1_loss(bg_rgb_pred, bg_rgb_label) * config.BG_WEIGHT
                loss = fg_loss + bg_loss
            
                self.optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()

                self.optimizer.step()

                if self.global_step % 10 == 0:
                    print(f'Process {self.rank} epoch {epoch} step {self.global_step} loss {loss.item()}')
                    self.logfn({"train/loss": loss.item()})
                
                self.global_step += 1

            current_lr = self.optimizer.param_groups[0]['lr']
            self.logfn({"learning_rate": current_lr})
            self.scheduler.step()   # Update the learning rate scheduler: "Training ImageNet in 1 Hour"

            if self.rank == 0:   # log only from the first process
                if (epoch > 0 and epoch % 10 == 0) or epoch == num_epochs - 1:   # original 10
                    print('**** VALIDATION AND PLOTTING ****')
                    self.val(split='train')
                    self.val(split='validation')

                    self.plot(split='train')
                    self.plot(split='validation')

                if epoch % 50 == 1:  # original 50
                    self.save_ckpt(all=True)
                    print('Saved checkpoint.')
                
                self.save_ckpt()
                print('Saved checkpoint.')

            dist.barrier(group=self.process_group)
                        
    @torch.no_grad()
    def val(self, split: str) -> None:
        print(f'Validation on {split} set...')
        
        loader = self.train_loader if split == "train" else self.val_loader

        self.weights2space.eval()

        psnrs = []
        psnrs_bg = []
        idx = 0

        for batch_idx, batch in enumerate(loader):

            train_nerf, mlp_weights, mlp_matrix, data_dir, background_indices, _ = batch
            rays = train_nerf['rays']
            color_bkgds = train_nerf['color_bkgd']
            color_bkgds = color_bkgds[0].unsqueeze(0).expand(len(mlp_matrix), -1)
            
            rays = rays._replace(origins=rays.origins.to(self.device), viewdirs=rays.viewdirs.to(self.device))
            color_bkgds = color_bkgds.to(self.device)
            mlp_matrix = mlp_matrix.to(self.device)
            #with autocast():
            pixels, alpha, filtered_rays, broken_batch_idxs = render_image_GT_shapenerf_objanerf(
                        radiance_field=self.ngp_mlp, 
                        occupancy_grid_shapenerf=self.occupancy_grid_shapenerf,
                        occupancy_grid_objanerf=self.occupancy_grid_objanerf, 
                        rays=rays, 
                        scene_aabb=self.scene_aabb, 
                        render_step_size=self.render_step_size,
                        color_bkgds=color_bkgds,
                        ngp_mlp_weights=mlp_weights,
                        device=self.device,
                        data_dirs=data_dir)

            pixels = pixels * alpha + color_bkgds.unsqueeze(1) * (1.0 - alpha)
            
            triplane = self.weights2space(mlp_matrix)
            
            rgb, _, _, _, bg_rgb_pred, bg_rgb_label = render_image_shapenerf_objanerf_triplane(
                self.weights2space.module.decoder,
                triplane,
                self.occupancy_grid_shapenerf,
                self.occupancy_grid_objanerf,
                filtered_rays,
                self.scene_aabb,
                render_step_size=self.render_step_size,
                render_bkgd=color_bkgds,
                background_indices=background_indices,
                max_foreground_coordinates=config.MAX_FOREGROUND_COORDINATES,
                max_background_coordinates=config.MAX_BACKGROUND_COORDINATES,
                device=self.device,
                data_dirs=data_dir
            )
            
            fg_mse = F.mse_loss(rgb, pixels) * config.FG_WEIGHT
            bg_mse = F.mse_loss(bg_rgb_pred, bg_rgb_label) * config.BG_WEIGHT

            mse_bg = fg_mse + bg_mse
            mse = F.mse_loss(rgb, pixels)
            
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())

            psnr_bg = -10.0 * torch.log(mse_bg) / np.log(10.0)
            psnrs_bg.append(psnr_bg.item())

            if idx > 99:
                break
            idx+=1

        mean_psnr = sum(psnrs) / len(psnrs)
        mean_psnr_bg = sum(psnrs_bg) / len(psnrs_bg)

        self.logfn({f'{split}/PSNR': mean_psnr})
        self.logfn({f'{split}/PSNR_BG': mean_psnr_bg})
        
        if split == 'validation' and mean_psnr > self.best_psnr:
            self.best_psnr = mean_psnr
            self.save_ckpt(best=True)
    
    @torch.no_grad()
    def plot(self, split: str) -> None:
        
        loader = self.train_loader if split == "train" else self.val_loader

        print(f'Plot on {split} set...')

        self.weights2space.eval()

        loader_iter = iter(loader)
        _, mlp_weights, mlp_matrix, data_dirs, _, _ = next(loader_iter)
        # since test_nerf is not present, rays and color_bkgds will be computed while rendering
        
        # must compute test_nerf rays and must pick only the first element of the batch
        data_dir = data_dirs[0]
        mlp_matrix = mlp_matrix[0].unsqueeze(0).to(self.device)
        mlp_weights = OrderedDict({'mlp_base.params': mlp_weights['mlp_base.params'][0].unsqueeze(0).to(self.device)})
        nerf_loader = ShapeNeRFObjaNeRFLoaderGT(
            data_dir=data_dir.replace('/mlp_128', ''),
            num_rays=config.NUM_RAYS,
            device=self.device,
            weights_file_name=config.NERF_WEIGHTS_FILE_NAME_OBJANERF if 'objaverse' in data_dir else config.NERF_WEIGHTS_FILE_NAME_SHAPENERF)
        
        nerf_loader.training = False
                
        # Getting just the first image in the dataset for performance reasons. In the future, we could use more elements.
        test_data = nerf_loader[0]  
        test_color_bkgd = test_data["color_bkgd"]
        test_rays = test_data["rays"]
        rays = Rays(origins=test_rays.origins.unsqueeze(0), viewdirs=test_rays.viewdirs.unsqueeze(0))
        color_bkgds = test_color_bkgd.unsqueeze(0)
        
        #with autocast():
        pixels, alpha, _, broken_batch_idxs = render_image_GT_shapenerf_objanerf(
                        radiance_field=self.ngp_mlp, 
                        occupancy_grid_shapenerf=self.occupancy_grid_shapenerf,
                        occupancy_grid_objanerf=self.occupancy_grid_objanerf,
                        rays=rays, 
                        scene_aabb=self.scene_aabb, 
                        render_step_size=self.render_step_size,
                        color_bkgds=color_bkgds,
                        ngp_mlp_weights=mlp_weights,
                        device=self.device,
                        training=False,
                        data_dirs=[data_dir])

        pixels = pixels * alpha + color_bkgds.unsqueeze(1).unsqueeze(1) * (1.0 - alpha)
    
        triplane = self.weights2space(mlp_matrix)

        print('**** len mlp matrix:', len(mlp_matrix))
        for idx in range(len(mlp_matrix)):
            data_dir = data_dirs[idx]
    
            rgb, _, _, _, _, _ = render_image_shapenerf_objanerf_triplane(
                self.weights2space.module.decoder,
                triplane=triplane,#[idx].unsqueeze(dim=0),
                occupancy_grid_shapenerf=self.occupancy_grid_shapenerf,
                occupancy_grid_objanerf=self.occupancy_grid_objanerf,
                rays=Rays(origins=rays.origins[idx].unsqueeze(dim=0), viewdirs=rays.viewdirs[idx].unsqueeze(dim=0)),
                scene_aabb=self.scene_aabb,
                render_step_size=self.render_step_size,
                render_bkgd=color_bkgds[idx].unsqueeze(dim=0),
                device=self.device,
                data_dirs=[data_dir]
            )
            
            rgb_A, alpha, _, _, _, _ = render_image_shapenerf_objanerf_triplane(
                            self.weights2space.module.decoder,
                            triplane=triplane,#[idx].unsqueeze(dim=0),
                            occupancy_grid_shapenerf=self.occupancy_grid_shapenerf,
                            occupancy_grid_objanerf=self.occupancy_grid_objanerf,
                            rays=Rays(origins=rays.origins[idx].unsqueeze(dim=0), viewdirs=rays.viewdirs[idx].unsqueeze(dim=0)),
                            scene_aabb=self.scene_aabb,
                            render_step_size=self.render_step_size,
                            render_bkgd=color_bkgds[idx].unsqueeze(dim=0),
                            device=self.device,
                            data_dirs=[data_dir]
            )
            
            gt_image = wandb.Image((pixels.to('cpu').detach().numpy()[idx] * 255).astype(np.uint8))
            pred_image_grid = wandb.Image((rgb.to('cpu').detach().numpy()[0] * 255).astype(np.uint8)) 
            pred_image_no_grid = wandb.Image((rgb_A.to('cpu').detach().numpy() * 255).astype(np.uint8))
            print('logging images...')
            self.logfn({f"{split}/nerf_{idx}": [gt_image, pred_image_grid, pred_image_no_grid]})

        print('Finished plotting.')
    
    def save_ckpt(self, best: bool = False, all: bool = False) -> None:
        print('Saving checkpoint...')
        ckpt = {
            "epoch": self.epoch,
            "weights2space": self.weights2space.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_psnr": self.best_psnr,
        }
        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()

        if all:
            ckpt_path = self.all_ckpts_path / f"{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)

        else:
            for previous_ckpt_path in self.ckpts_path.glob("*.pt"):
                if "best" not in previous_ckpt_path.name:
                    previous_ckpt_path.unlink()

            ckpt_path = self.ckpts_path / f"{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)

        if best:
            ckpt_path = self.ckpts_path / "best.pt"
            torch.save(ckpt, ckpt_path)
    
    
    def restore_from_last_ckpt(self) -> None:
        if self.ckpts_path.exists():
            ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "best" not in p.name]
            if len(ckpt_paths) > 0 : # if at least a checkpoint exists, load the latest o
                ckpt_path = get_latest_checkpoints_path(self.ckpts_path)
                print(f'loading weights: {ckpt_path}')
                ckpt = torch.load(ckpt_path, map_location='cpu')

                self.epoch = ckpt["epoch"] + 1
                print('new first epoch: ', self.epoch)
                self.global_step = self.epoch * len(self.train_loader)
                self.best_psnr = ckpt["best_psnr"]

                self.weights2space.load_state_dict(ckpt["weights2space"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                
                if "scheduler" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler"])
    
    def config_wandb(self, run_name, run_id=None):
        wandb.login()
        wandb_config = config.WANDB_CONFIG
        wandb_config["dataset"] = "shapenerf_objanerf"
        wandb.init(
            entity='andrea_amaduzzi',
            project='nerf2vec',
            name=run_name,
            config=wandb_config,
            id=run_id,  # run_id is used to resume training
            resume='must' if run_id is not None else 'never'
        )

    def linear_warmup_scheduler(self, optimizer, warmup_epochs, start_lr, target_lr):
        print('start_lr: ', start_lr)
        print('target_lr: ', target_lr)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (1 + (epoch / warmup_epochs) * (target_lr / start_lr - 1)) / self.world_size 
            return (target_lr / start_lr) / self.world_size
        return LambdaLR(optimizer, lr_lambda)


def main(rank, world_size, run_name, run_id=None):
    setup(rank, world_size)
    gpu_id = rank % torch.cuda.device_count()
    device = torch.device(f'cuda:{gpu_id}')   # must be ajusted for multi-node case
    torch.cuda.set_device(gpu_id)
    print('set device to: ', gpu_id)
    trainer = ShapeNeRFObjaNeRFTrainerParallel(rank=rank, world_size=world_size, device=device, run_name=run_name, run_id=run_id) 

    # Launch training
    trainer.train()
    cleanup()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    run_name = f'weights2space_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'    # use this run_name to begin training from scratch
    run_id = None
    #run_name =                                          # use this run_name to resume training. This makes sure that the last checkpoint is loaded correctly
    #run_id = 's3lbkh5o'                                 # use this line to resume run in wandb: the run_id is taken from wandb overview project page. When syncing, you must add the arg --append
    print('world_size:', world_size)
    print('rank:', rank)
    print('run_name:', run_name)
    main(rank, world_size, run_name, run_id)
