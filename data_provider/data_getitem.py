# coding : utf-8
# Author : yuxiang Zeng
# 根据场景需要来改这里的input形状
from torch.utils.data import Dataset
import numpy as np
import os
from baselines.narformer import tokenizer
from data_provider.create_latency import *
from scipy.sparse import csr_matrix
import dgl
from tqdm import *
import pickle 



class SeqDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.mode = mode
        self.x = x
        temp = []
        dx = dr = dp = 32
        os.makedirs('./datasets/nasbench201/split_dataset', exist_ok=True)
        filedir = f'./datasets/nasbench201/split_dataset/D{config.dataset}_S{config.spliter_ratio}_M{mode}_round_{config.runid}.pkl'
        try:
            with open(filedir, 'rb') as f:
                data = pickle.load(f)
                self.x = data['x']
                self.y = data['y']
        except:
            temp = []
            for i in trange(len(x)):
                key = x[i]
                arch_str = get_arch_str_from_arch_vector(key)             # 架构向量转字符串
                adj_mat, ops_idx = info2mat(arch_str)                     # 得到邻接矩阵与操作
                tokens = tokenizer(ops_idx, adj_mat, dx, dr, dp, 'nerf')  # 得到token表示
                temp.append(tokens)

            self.x = np.stack(temp)
            self.y = y

            with open(filedir, 'wb') as f:
                pickle.dump({'x': self.x, 'y': self.y}, f)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        tokens = self.x[idx]
        y = self.y[idx]
        return tokens, y

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        tokens, y = zip(*batch)
        tokens, y =  default_collate(tokens), default_collate(y)
        return tokens, y
    

class RNNDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.mode = mode
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        x = torch.as_tensor(self.x[idx], dtype=torch.long)
        y = self.y[idx]
        return x, y

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        x, y = zip(*batch)
        x, y =  default_collate(x), default_collate(y)
        return x, y
    
    
class ProxyDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode
        with open('./datasets/all_flops_parameter.pkl', 'rb') as f:
            self.proxy = pickle.load(f)
            
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].astype(np.int32)
        proxy = torch.tensor(self.proxy[tuple(x)], dtype=torch.float32) # [1, 2]
        if self.config.model == 'flops-mac':
            proxy = proxy[:]
        elif self.config.model == 'flops':
            proxy = proxy[0]
        y = self.y[idx]
        return proxy, y


    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        features, y = zip(*batch)
        # x, y = default_collate(x), default_collate(y)
        features, y = default_collate(features), default_collate(y)
        return features, y


class BRPNASDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        key, value = self.x[idx], self.y[idx]
        value = torch.tensor(value).float()
        graph, label = get_matrix_and_ops(key)
        graph, features = get_adjacency_and_features(graph, label)
        graph = torch.tensor(graph).float()
        features = torch.tensor(features).float()
        return graph, features, value

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        graph, features, y = zip(*batch)
        # x, y = default_collate(x), default_collate(y)
        graph, features, y = default_collate(graph), default_collate(features), default_collate(y)
        return graph, features, y
    
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