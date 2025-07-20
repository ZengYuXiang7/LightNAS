# coding : utf-8
# Author : yuxiang Zeng
# 根据场景需要来改这里的input形状
from torch.utils.data import Dataset
import numpy as np

from baselines.narformer import tokenizer
from data_provider.create_latency import *
from scipy.sparse import csr_matrix
import dgl



class SeqDataset(Dataset):
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
        tokens = self.get_token(x)
        return tokens, y

    def get_token(self, key):
        graph, _ = get_matrix_and_ops(key)
        arch_str = get_arch_str_from_arch_vector(key)
        adj_mat, ops_idx = info2mat(arch_str)
        dx = dr = dp = 32
        tokens = tokenizer(ops_idx, adj_mat, dx, dr, dp, 'nerf')
        return tokens

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        tokens, y = zip(*batch)
        tokens, y =  default_collate(tokens), default_collate(y)
        return tokens, y
    
    
class GraphDataset(Dataset):
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
        return graph, op_idx, y

    def get_graph(self, key):
        graph, one_hot_idx = get_matrix_and_ops(key)
        graph, op_idx = get_adjacency_and_features(graph, one_hot_idx)
        graph = dgl.from_scipy(csr_matrix(graph))
        graph = dgl.to_bidirected(graph)
        features = torch.tensor(op_idx).long()
        op_idx = torch.argmax(features, dim=1)
        return graph, op_idx

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        graph, op_idx, y = zip(*batch)
        # x, y = default_collate(x), default_collate(y)
        graph, op_idx, y = dgl.batch(graph), default_collate(op_idx), default_collate(y)
        return graph, op_idx, y



class TensorDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode

    def __len__(self):
        # return len(self.x)
        return len(self.x) - self.config.seq_len - self.config.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.config.seq_len
        r_begin = s_end
        r_end = r_begin + self.config.pred_len

        if not self.config.multi_dataset:
            x = self.x[s_begin:s_end][:, -3:]
            x_fund = self.x[s_begin:s_end][:, 0]
            x_mark = self.x[s_begin:s_end][:, :-1] if not self.config.dataset == 'financial' else self.x[s_begin:s_end][:, 1:-1]
            y = self.y[r_begin:r_end]
        else:
            x = self.x[s_begin:s_end][:, :, -3:]
            x_fund = self.x[s_begin:s_end][:, :, 0]
            x_mark = self.x[s_begin:s_end][:, :, -1] if not self.config.dataset == 'financial' else self.x[s_begin:s_end][:, :, 1:-1]
            y = self.y[r_begin:r_end]

        # print(x.shape, y.shape)
        return x, x_mark, x_fund, y

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        x, x_mark, x_fund, y = zip(*batch)
        x, y = default_collate(x), default_collate(y)
        x_mark = default_collate(x_mark)
        x_fund = default_collate(x_fund).long()
        return x, x_mark, x_fund, y


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode

    def __len__(self):
        # return len(self.x)
        return len(self.x) - self.config.seq_len - self.config.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.config.seq_len
        r_begin = s_end
        r_end = r_begin + self.config.pred_len

        x = self.x[s_begin:s_end][:, 4:]
        x_mark = self.x[s_begin:s_end][:, :4]
        y = self.y[r_begin:r_end]
        return x, x_mark, y

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        x, x_mark, y = zip(*batch)
        x, y = default_collate(x), default_collate(y)
        x_mark = default_collate(x_mark)
        return x, x_mark, y