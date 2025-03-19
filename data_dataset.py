# coding : utf-8
# Author : yuxiang Zeng
# 根据场景需要来改这里的input形状
import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
import dgl
import torch

from modules.load_data.create_latency import get_matrix_and_ops, get_adjacency_and_features


class TensorDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        graph, op_idx = self.get_graph(x)
        # x, y = torch.from_numpy(x), torch.from_numpy(y)
        return graph, op_idx, y

    def get_graph(self, key):
        graph, one_hot_idx = get_matrix_and_ops(key)
        graph, op_idx = get_adjacency_and_features(graph, one_hot_idx)
        graph = dgl.from_scipy(csr_matrix(graph))
        graph = dgl.to_bidirected(graph)
        features = torch.tensor(op_idx).long()
        op_idx = torch.argmax(features, dim=1)
        return graph, op_idx

def custom_collate_fn(batch, config):
    from torch.utils.data.dataloader import default_collate
    graph, op_idx, y = zip(*batch)
    # x, y = default_collate(x), default_collate(y)
    graph, op_idx, y = dgl.batch(graph), default_collate(op_idx), default_collate(y)
    return graph, op_idx, y