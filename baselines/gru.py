# coding : utf-8
# Author : yuxiang Zeng

import torch

class GRU(torch.nn.Module):
    def __init__(self, input_dim, config):
        super(GRU, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.transfer = torch.nn.Linear(input_dim * 5, self.d_model)
        self.gru = torch.nn.GRU(self.d_model, self.d_model, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(self.d_model * 6, 1)

    def forward(self, features):
        features = torch.nn.functional.one_hot(features, 5).float()
        x = self.transfer(features)
        out, _ = self.gru(x)
        out = out.reshape(x.shape[0], -1)
        y = self.fc(out)
        return y


