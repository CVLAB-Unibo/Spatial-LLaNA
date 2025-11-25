'''

from nf2vec 

'''

from typing import Callable, Tuple, List, Union

import torch
from einops import repeat
from torch import Tensor, nn
import tinycudann as tcnn
import sys
sys.path.append('../nf2vec')
from nerf.instant_ngp import _TruncExp

class CoordsEncoder:
    def __init__(
        self,
        encoding_conf: dict,
        input_dims: int = 3,
        device: torch.device=torch.device('cuda:0')
    ) -> None:
        self.input_dims = input_dims

        self.coords_enc = tcnn.Encoding(input_dims, encoding_conf, seed=999).to(device)
        self.out_dim = self.coords_enc.n_output_dims

    def apply_encoding(self, x):
        self.coords_enc = self.coords_enc.to(x.device)
        return self.coords_enc(x)

    def embed(self, inputs: Tensor) -> Tensor:
        # return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        result_encoding = self.apply_encoding(inputs.view(-1, 3))
        result_encoding = result_encoding.view(inputs.size()[0],inputs.size()[1],-1)
        return result_encoding

class ImplicitDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers_before_skip: int,
        num_hidden_layers_after_skip: int,
        out_dim: int,
        encoding_conf: dict,  # Added for NerfAcc
        aabb: Union[torch.Tensor, List[float]],  # Added for NerfAcc
        triplane_hidden_dim: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.coords_enc = CoordsEncoder(encoding_conf=encoding_conf, input_dims=in_dim, device=device)
        coords_dim = self.coords_enc.out_dim

        # ################################################################################
        # Added for NerfAcc
        # ################################################################################
        trunc_exp = _TruncExp.apply
        self.density_activation = lambda x: trunc_exp(x - 1)
        self.aabb = aabb
        self.in_dim = in_dim
        # ################################################################################

        self.in_layer = nn.Sequential(nn.Linear(triplane_hidden_dim + coords_dim, hidden_dim), nn.ReLU())

        self.skip_proj = nn.Sequential(nn.Linear(triplane_hidden_dim + coords_dim, hidden_dim), nn.ReLU())

        before_skip = []
        for _ in range(num_hidden_layers_before_skip):
            before_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.before_skip = nn.Sequential(*before_skip)

        after_skip = []
        for _ in range(num_hidden_layers_after_skip):
            after_skip.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        after_skip.append(nn.Linear(hidden_dim, out_dim))
        self.after_skip = nn.Sequential(*after_skip)
        
        self.triplane_hidden_dim = triplane_hidden_dim

    def forward(self, triplane: Tensor, coords: Tensor) -> Tensor:
        # triplane: (Bx3, 768, 32, 32)
        # coords: (B, N, 3)
        # Sometimes the ray march algorithm calls the model with an input with 0 length.
        # The CutlassMLP crashes in these cases, therefore this fix has been applied.
        batch_size, n_coords, _ = coords.size()
        if n_coords == 0:
            rgb = torch.zeros([batch_size, 0, 3], device=coords.device)
            density = torch.zeros([batch_size, 0, 1], device=coords.device)
            return rgb, density

        # ################################################################################
        # Added for NerfAcc
        # ################################################################################
        aabb_min, aabb_max = torch.split(self.aabb, self.in_dim, dim=-1)
        scaled_coords = coords.clone()
        scaled_coords = 2*(scaled_coords - aabb_min) / (aabb_max - aabb_min) - 1    # normalizes coords to [-1, 1] range
        coords = (coords - aabb_min) / (aabb_max - aabb_min)                        # normalizes coords to [0, 1] range
        
        selector = ((coords > 0.0) & (coords < 1.0)).all(dim=-1)
        # ################################################################################
        
        # run interpolation inside the triplane, to get features
        # here, unlike the original code in nerfacc_nerf2vec/examples/radiance_fields/ngp_nerf2vec.py, we worked with batched coords
        coords_xy = torch.cat((scaled_coords[:, :, 0].unsqueeze(2), scaled_coords[:, :, 1].unsqueeze(2)), dim=2)    # (B, N, 2)
        coords_xz = torch.cat((scaled_coords[:, :, 0].unsqueeze(2), scaled_coords[:, :, 2].unsqueeze(2)), dim=2)    # (B, N, 2)
        coords_zy = torch.cat((scaled_coords[:, :, 1].unsqueeze(2), scaled_coords[:, :, 2].unsqueeze(2)), dim=2)    # (B, N, 2)
        grid = torch.stack([coords_xy, coords_xz, coords_zy], dim=1)
        grid = grid.reshape(-1, n_coords, 2).unsqueeze(1)   # (Bx3, 1, N, 2)
        features_sample = torch.nn.functional.grid_sample(triplane, grid, align_corners=True).squeeze(2).permute(0, 2, 1)   # (Bx3, N, triplane_hidden_dim)
        features_sample = features_sample.reshape(batch_size, 3, n_coords, self.triplane_hidden_dim)
        features_sample = features_sample.sum(1)
        
        # concatenate encoded coords with triplane features: treat this tensor as input to the decoder mlp
        feats_and_coords = torch.cat([self.coords_enc.embed(coords),features_sample], dim=-1)   # (B, N, 144+16)
        x = self.in_layer(feats_and_coords)
        x = self.before_skip(x)

        inp_proj = self.skip_proj(feats_and_coords)
        x = x + inp_proj

        x = self.after_skip(x)  # (1, 35000, 4)
        # return x.squeeze(-1) # ORIGINAL INR2VEC IMPLEMENTATION

        # ################################################################################
        # Added for NerfAcc
        # ################################################################################
        rgb, density_before_activation = x[..., :3], x[..., 3]
        density_before_activation = density_before_activation[:, :, None]

        # Be sure that the density is non-negative
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )

        rgb = torch.nn.Sigmoid()(rgb)

        return rgb, density
        # ################################################################################