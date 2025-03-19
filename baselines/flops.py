# coding : utf-8
# Author : yuxiang Zeng
import pickle
import torch


class Flops(torch.nn.Module):
    def __init__(self, args):
        super(Flops, self).__init__()
        self.encoder = torch.nn.Linear(5 + 1, args.rank)
        with open('./baslines/flops.pkl', 'rb') as f:
            self.flops = pickle.load(f)
        self.linear = torch.nn.Linear(args.rank, 1)

    def forward(self, key):
        inputs = torch.cat(key, self.flops[key])
        embeds = self.encoder(inputs)
        y = self.linear(embeds)
        return y