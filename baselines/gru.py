# coding : utf-8
# Author : yuxiang Zeng

import torch

class GRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GRU, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.transfer = torch.nn.Linear(input_dim * 5, hidden_dim)
        self.lstm = torch.nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_dim, output_dim)

    def forward(self, _, features):
        features = torch.nn.functional.one_hot(features, 5).float()
        features = features.reshape(features.shape[0], -1)
        x = self.transfer(features)
        out, op_embeds = self.lstm(x)
        y = self.fc(out)
        return y

