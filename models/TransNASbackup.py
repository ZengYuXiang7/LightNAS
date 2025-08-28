
# coding : utf-8
# Author : Yuxiang Zeng
import random
import torch
from baselines.narformer import *
from models.layers.transformer import Transformer
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
            dim_feedforward=self.d_model * 2,
            batch_first=True,
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # 输出层（cls token → 预测值）
        self.fc = nn.Linear(self.d_model, 1)

        self.dropout = nn.Dropout(0.10)

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
    
    
    # def transfer(self, x)
# 
        # return y
        


class PairwiseDiffLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        """
        :param loss_type: 'l1', 'l2', or 'kldiv'
        """
        super(PairwiseDiffLoss, self).__init__()
        loss_type = loss_type.lower()
        if loss_type == 'l1':
            self.base_loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'kldiv':
            self.base_loss = nn.KLDivLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, predicts, targets):
        """
        :param predicts: Tensor of shape (B,) or (B, 1)
        :param targets: Tensor of shape (B,) or (B, 1)
        :return: Pairwise difference loss
        """
        # 自动 squeeze 支持 [B, 1] 输入
        if predicts.ndim == 2 and predicts.shape[1] == 1:
            predicts = predicts.squeeze(1)
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        if predicts.ndim != 1 or targets.ndim != 1:
            raise ValueError("Both predicts and targets must be 1D tensors.")

        B = predicts.size(0)
        idx = list(range(B))
        random.shuffle(idx)
        shuffled_preds = predicts[idx]
        shuffled_targs = targets[idx]

        diff_preds = predicts - shuffled_preds
        diff_targs = targets - shuffled_targs

        return self.base_loss(diff_preds, diff_targs)
    
    
    
class ACLoss(nn.Module):
    def __init__(self, loss_type='l1', reduction='mean'):
        """
        Architecture Consistency Loss
        :param loss_type: 'l1' or 'l2'
        :param reduction: 'mean' or 'sum'
        """
        super(ACLoss, self).__init__()
        if loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, predictions):
        """
        :param predictions: Tensor of shape [2B, 1] or [2B]
                            where front B are source, back B are augmented
        :return: scalar loss
        """
        N = predictions.shape[0]
        B = N // 2  # 自动 floor 向下取整

        # 仅使用前 B 和后 B，丢弃中间多出的一个（若存在）
        source = predictions[:B]
        augmented = predictions[-B:]
        
        
        # Ensure shape is [B]
        source = source.squeeze(-1) if source.ndim == 2 and source.shape[1] == 1 else source
        augmented = augmented.squeeze(-1) if augmented.ndim == 2 and augmented.shape[1] == 1 else augmented

        return self.criterion(source, augmented)