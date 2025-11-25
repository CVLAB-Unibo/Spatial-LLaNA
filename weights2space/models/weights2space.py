from torch import nn
from typing import List
import torch
from models.encoder import Encoder
import sys
sys.path.append('../nf2vec')
#from nerf.utils import Rays, render_image_shapenerf_objanerf, render_image_GT_shapenerf_objanerf
import yaml
from models.transformer import nf2vecTransformerDecoderLayer
from models.idecoder import ImplicitDecoder


class Weights2Space(nn.Module):
    def __init__(self, device: torch.device, cfg) -> None:
        super().__init__()
        self.device = device
        self.encoder = Encoder(cfg.MLP_UNITS, cfg.ENCODER_HIDDEN_DIM, cfg.ENCODER_EMBEDDING_DIM)
        self.nf2vec_width = cfg.ENCODER_EMBEDDING_DIM   # nf2vec provides 336 tokens of 1024 dimensions
        
        decoder_layer = nf2vecTransformerDecoderLayer(batch_first=True, d_model=cfg.LEARNABLE_QUERY_DIM, nhead=cfg.N_HEADS, dim_feedforward=cfg.DIM_FEEDFORWARD, norm_first=True, kdim=self.nf2vec_width, vdim=self.nf2vec_width)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.N_LAYERS)
        self.query_tokens = nn.Parameter(torch.zeros(1, cfg.N_LEARNABLE_QUERY, cfg.LEARNABLE_QUERY_DIM))
        self.query_tokens.data.normal_(mean=0.0, std=cfg.QUERY_INIT_STD)
        self.decoder = ImplicitDecoder(
            in_dim=cfg.DECODER_INPUT_DIM,
            hidden_dim=cfg.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=cfg.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=cfg.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=cfg.DECODER_OUT_DIM,
            encoding_conf=cfg.INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(cfg.GRID_AABB, dtype=torch.float32, device=self.device),
            triplane_hidden_dim=cfg.TRIPLANE_PARAMS['hidden_dim'],
            device=device,
        )
        
        self.triplane_params = cfg.TRIPLANE_PARAMS
        self.encoder.to(device)
        self.transformer.to(device)
        self.decoder.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)                       
        nf2vec_embeds = self.encoder(x)                                                                     
        nf2vec_embeds = nf2vec_embeds.permute(0, 2, 1)
        query_tokens = self.query_tokens.expand(x.size(0), -1, -1).to(self.device) 
        query_output = self.transformer(
            tgt=query_tokens,
            memory=nf2vec_embeds) 

        # correct reshape to get 516-dim spatial token, when we have 3072x516 tokens
        features_reshaped = query_output.reshape(query_output.shape[0], 3, self.triplane_params['resolution'], self.triplane_params['resolution'], self.triplane_params['hidden_dim'])
        features_permuted = features_reshaped.permute(0, 1, 4, 2, 3)
        triplane = features_permuted.reshape(query_output.shape[0]*3, self.triplane_params['hidden_dim'], self.triplane_params['resolution'], self.triplane_params['resolution'])   
        return triplane