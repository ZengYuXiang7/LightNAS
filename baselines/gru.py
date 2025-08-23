# coding : utf-8
# Author : yuxiang Zeng

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, input_dim, config):
        super(GRU, self).__init__()
        self.config = config
        self.d_model = config.d_model

        self.num_classes = 29 if config.dataset == 'nnlqp' else 6
        self.transfer = nn.Linear(self.num_classes, self.d_model)
        self.gru = nn.GRU(self.d_model, self.d_model, num_layers=1, batch_first=True)

        self.attn = nn.Linear(self.d_model, 1)
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1)
        )

    def forward(self, features):
        # features: [B, T, input_dim]
        if self.config.dataset != 'nnlqp':
            features = F.one_hot(features.long(), num_classes=self.num_classes).float()
        # -> [B, T, input_dim, num_classes]
        features = features.view(features.size(0), features.size(1), -1)  # Flatten input_dim*num_classes

        x = self.transfer(features)     # [B, T, d_model]
        out, _ = self.gru(x)            # [B, T, d_model]

        attn_score = self.attn(out)                 # [B, T, 1]
        attn_weights = F.softmax(attn_score, dim=1) # [B, T, 1]
        pooled = torch.sum(attn_weights * out, dim=1)  # [B, d_model]

        y = self.fc(pooled)                             # [B, 1]
        return y