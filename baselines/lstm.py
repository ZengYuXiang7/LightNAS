# coding : utf-8
# Author : yuxiang Zeng

import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, config):
        super(LSTM, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.transfer = torch.nn.Linear(input_dim * 5, self.d_model)
        self.lstm = torch.nn.LSTM(self.d_model, self.d_model, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(self.d_model * 6, 1)

    def forward(self, features):
        features = torch.nn.functional.one_hot(features, 5).float()
        x = self.transfer(features)
        out, (hn, cn) = self.lstm(x)
        out = out.reshape(x.shape[0], -1)
        y = self.fc(out)
        return y
