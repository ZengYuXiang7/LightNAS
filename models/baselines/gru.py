# coding : utf-8
# Author : Anonymous

import torch

class GRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GRU, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.transfer = torch.nn.Linear(input_dim * 7, hidden_dim)
        self.lstm = torch.nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_dim, output_dim)

    def forward(self, _, features):
        features = torch.nn.functional.one_hot(features, 7).float()
        features = features.reshape(features.shape[0], -1)
        x = self.transfer(features)
        # 每个架构是独立样本，不把批次维当成时间序列。
        x = x.unsqueeze(1)
        out, op_embeds = self.lstm(x)
        y = self.fc(out[:, 0, :])
        return y

