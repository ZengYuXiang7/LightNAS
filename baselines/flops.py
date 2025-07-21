# coding : utf-8
# Author : yuxiang Zeng
import pickle
import torch

class Flops(torch.nn.Module):
    def __init__(self, input_size, config):
        super(Flops, self).__init__()
        
        # 第一层线性变换，输入2维数据，输出d_model维数据
        self.encoder = torch.nn.Linear(input_size, config.d_model)
        # Backbone 两层 MLP, 包含激活函数（ReLU）和归一化层（BatchNorm）
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, config.d_model),  # 第一层线性层
            torch.nn.LayerNorm(config.d_model),  # 第一层 BatchNorm 归一化
            torch.nn.ReLU(),  # ReLU 激活函数
            torch.nn.Linear(config.d_model, config.d_model),  # 第二层线性层
            torch.nn.LayerNorm(config.d_model),  # 第二层 BatchNorm 归一化
            torch.nn.ReLU(),  # ReLU 激活函数
        )
        # 最后的输出层
        self.decoder = torch.nn.Linear(config.d_model, 1)

    def forward(self, features):
        if len(features.shape) == 1:
            features = features.unsqueeze(-1)
        embeds = self.encoder(features)  # 第一层线性变换
        hidden = self.backbone(embeds)  # Backbone 两层 MLP 网络
        y = self.decoder(hidden)  # 输出层
        return y
