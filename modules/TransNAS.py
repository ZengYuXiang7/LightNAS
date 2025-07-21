# coding : utf-8
# Author : Yuxiang Zeng
import torch

import einops
from baselines.narformer import *
from layers.transformer import Transformer
# self.encoder = Transformer(config.d_model, config.num_heads, config.num_layers, 'rms', 'ffn', 'self')

import torch
import torch.nn as nn
import einops

class TransNAS(nn.Module):
    def __init__(self, enc_in, config):
        super(TransNAS, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.seq_len = 9  # 如果是 9 个 token（例如来自 NAS cell）

        # 输入 embedding 映射
        self.enc_embedding = nn.Linear(enc_in, self.d_model)

        # 可学习的 cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # 可学习的位置编码：cls_token + patch_token
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len + 1, self.d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.num_heads,
            dim_feedforward=self.d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # 输出层（cls token → 预测值）
        self.fc = nn.Linear(self.d_model, 1)

        self.dropout = nn.Dropout(config.dropout)

        # 初始化
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x.shape: [B, seq_len, enc_in]
        B, _, _ = x.shape

        x = self.enc_embedding(x)  # [B, seq_len, d_model]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, seq_len+1, d_model]
        x = x + self.pos_embedding                     # 加位置编码

        x = self.dropout(x)
        x = self.encoder(x)                            # 经过 transformer encoder

        cls_out = x[:, 0]                              # 取 cls_token 的输出
        y = self.fc(cls_out)                           # 回归或分类

        return y
