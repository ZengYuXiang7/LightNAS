####################################################################################################
# HELP: hardware-adaptive efficient latency prediction for nas via meta-learning, NeurIPS 2021
# Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang
# github: https://github.com/HayeonLee/HELP, email: hayeon926@kaist.ac.kr
####################################################################################################
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import math
from utils import *



class HELPBase(nn.Module):
    """
    The base model for MAML (Meta-SGD) for meta-NAS-predictor.
    """

    def __init__(self, nfeat, args):
        super(HELPBase, self).__init__()
        self.hw_embed_on = False
        self.layer_size = 100
        hw_embed_dim = 100
        for i in range(1, 5):
            if i == 1:
                input_dim = nfeat
            else:
                input_dim = self.layer_size
            self.add_module(f'gc{i}', GraphConvolution(input_dim, self.layer_size))
        if self.hw_embed_on:
            self.add_module('fc_hw1', nn.Linear(hw_embed_dim, self.layer_size))
            self.add_module('fc_hw2', nn.Linear(self.layer_size, self.layer_size))
            hfeat = self.layer_size * 2
        else:
            hfeat = self.layer_size

        self.add_module('fc3', nn.Linear(hfeat, hfeat))
        self.add_module('fc4', nn.Linear(hfeat, hfeat))

        self.add_module('fc5', nn.Linear(hfeat, 1))
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        init.uniform_(self.gc1.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc2.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc3.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc4.weight, a=-0.05, b=0.05)


    def forward(self, adj, feat, hw_embed=None, params=None):
        assert len(feat) == len(adj)
        if params == None:
            out = self.relu(self.gc1(feat, adj).transpose(2, 1))
            out = out.transpose(1, 2)
            out = self.relu(self.gc2(out, adj).transpose(2, 1))
            out = out.transpose(1, 2)
            out = self.relu(self.gc3(out, adj).transpose(2, 1))
            out = out.transpose(1, 2)
            out = self.relu(self.gc4(out, adj).transpose(2, 1))
            out = out.transpose(1, 2)
            out = out[:, out.size()[1] - 1, :]

            if self.hw_embed_on:
                hw_embed = hw_embed.repeat(len(feat), 1)
                hw = self.relu(self.fc_hw1(hw_embed))
                hw = self.relu(self.fc_hw2(hw))
                out = torch.cat([out, hw], dim=-1)

            out = self.relu(self.fc3(out))
            out = self.relu(self.fc4(out))
            out = self.fc5(out)

        else:
            out = F.relu(self.gc1(feat, adj, weight=params['meta_learner.gc1.weight'],
                                  bias=params['meta_learner.gc1.bias']).transpose(2, 1))
            out = out.transpose(1, 2)
            out = F.relu(self.gc2(out, adj, weight=params['meta_learner.gc2.weight'],
                                  bias=params['meta_learner.gc2.bias']).transpose(2, 1))
            out = out.transpose(1, 2)
            out = F.relu(self.gc3(out, adj, weight=params['meta_learner.gc3.weight'],
                                  bias=params['meta_learner.gc3.bias']).transpose(2, 1))
            out = out.transpose(1, 2)
            out = F.relu(self.gc4(out, adj, weight=params['meta_learner.gc4.weight'],
                                  bias=params['meta_learner.gc4.bias']).transpose(2, 1))
            out = out.transpose(1, 2)
            out = out[:, out.size()[1] - 1, :]

            if self.hw_embed_on:
                hw_embed = hw_embed.repeat(len(feat), 1)

                hw = F.relu(F.linear(hw_embed, params['meta_learner.fc_hw1.weight'],
                                     params['meta_learner.fc_hw1.bias']))
                hw = F.relu(F.linear(hw, params['meta_learner.fc_hw2.weight'],
                                     params['meta_learner.fc_hw2.bias']))
                out = torch.cat([out, hw], dim=-1)

            out = F.relu(F.linear(out, params['meta_learner.fc3.weight'],
                                  params['meta_learner.fc3.bias']))
            out = F.relu(F.linear(out, params['meta_learner.fc4.weight'],
                                  params['meta_learner.fc4.bias']))
            out = F.linear(out, params['meta_learner.fc5.weight'],
                           params['meta_learner.fc5.bias'])
        return out


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_, adj, weight=None, bias=None):
        if weight is not None:
            support = torch.matmul(input_, weight)
            output = torch.bmm(adj, support)
            if bias is not None:
                return output + bias
            else:
                return output

        else:
            support = torch.matmul(input_, self.weight)
            output = torch.bmm(adj, support)
            if self.bias is not None:
                return output + self.bias
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
