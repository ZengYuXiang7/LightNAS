# coding : utf-8
# Author : Yuxiang Zeng
import torch

import einops
from baselines.narformer import *

class TransNAS(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(TransNAS, self).__init__()
        self.config = config
        self.d_model = config.d_model
        
        # self.enc_embedding = torch.nn.Linear(enc_in, config.d_model)
        # self.encoder = Transformer(config.d_model, config.num_heads, config.num_layers, 'rms', 'ffn', 'self')
        # self.fusion = torch.nn.Linear(config.d_model * 9, config.d_model)
        # self.fc = torch.nn.Linear(config.d_model, 1)
        self.transformer = Encoder(config)
        self.mlp = RegHead(config)
        
    def forward(self, x):
        # x_enc = self.enc_embedding(x)
        # x_enc = self.encoder(x_enc)
        # x_enc = einops.rearrange(x_enc, 'bs seq d -> bs (seq d)')
        # x_enc = self.fusion(x_enc)
        # y = self.fc(x_enc)
        x_enc = self.transformer(x) #multi_stage:aev(b, 1, d)
        y = self.mlp(x_enc, None)
        return y
