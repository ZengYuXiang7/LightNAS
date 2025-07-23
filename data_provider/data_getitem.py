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
from dgl import from_scipy


    
class ProxyDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode
        if config.dataset == 'nasbench201':
            with open('./datasets/nasbench201_all_flops_parameter.pkl', 'rb') as f:
                self.proxy = pickle.load(f)
        elif config.dataset == 'nnlqp':
            with open('./datasets/nnlqp_all_flops_parameter.pkl', 'rb') as f:
                self.proxy = pickle.load(f)
            
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].astype(np.int32)

        if self.config.dataset == 'nasbench201':
            proxy = torch.tensor(self.proxy[tuple(x)], dtype=torch.float32) # [1, 2]
        elif self.config.dataset == 'nnlqp':
            proxy = torch.tensor(self.proxy[x], dtype=torch.float32) # [1, 2]
            
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




class RNNDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.mode = mode
        self.x = x
        self.y = y
        if config.dataset == 'nnlqp':
            with open('./datasets/nnlqp/unseen_structure/graph/all_features.pkl', 'rb') as f:
                self.all_features = pickle.load(f)
                
    def __len__(self):
        return len(self.x)

    def get_bench_201(self, features):
        features = features.astype(np.int32)
        return features
    
    def get_nnlqp(self, key):
        key = key.astype(np.int32) - 1
        features = self.all_features[key]            # np.array or list
        # features = np.argmax(features)
        # print(features)
        return features
    
    def __getitem__(self, idx):
        key, value = self.x[idx], self.y[idx]
        value = torch.tensor(value).float()
        if self.config.dataset == 'nasbench201':
            features = self.get_bench_201(key)
        elif self.config.dataset == 'nnlqp':
            features = self.get_nnlqp(key)
        return features, value

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        x, y = zip(*batch)
        x, y =  default_collate(x), default_collate(y)
        return x, y
    


class BRPNASDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode
        
        if config.dataset == 'nnlqp':
            # 读取全部图和特征
            with open('./datasets/nnlqp/unseen_structure/graph/all_graphs.pkl', 'rb') as f:
                self.all_graphs = pickle.load(f)
            with open('./datasets/nnlqp/unseen_structure/graph/all_features.pkl', 'rb') as f:
                self.all_features = pickle.load(f)

    def __len__(self):
        return len(self.x)
    
    def get_bench_201(self, key):
        graph, label = get_matrix_and_ops(key)
        graph, features = get_adjacency_and_features(graph, label)
        graph = torch.tensor(graph).float()
        features = torch.tensor(features).float()
        return graph, features
    
    def get_nnlqp(self, key):
        key = key.astype(np.int32) - 1
        graph = self.all_graphs[key]             # scipy sparse matrix
        features = self.all_features[key]            # np.array or list
        return graph, features
    
    def __getitem__(self, idx):
        key, value = self.x[idx], self.y[idx]
        value = torch.tensor(value).float()
        if self.config.dataset == 'nasbench201':
            graph, features = self.get_bench_201(key)
        elif self.config.dataset == 'nnlqp':
            graph, features = self.get_nnlqp(key)
        return graph, features, value


    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        graph, features, y = zip(*batch)
        graph, features, y = default_collate(graph), default_collate(features), default_collate(y)
        return graph, features, y



class GraphDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode
        
        if config.dataset == 'nnlqp':
            # 读取全部图和特征
            with open('./datasets/nnlqp/unseen_structure/graph/all_graphs.pkl', 'rb') as f:
                self.all_graphs = pickle.load(f)
            with open('./datasets/nnlqp/unseen_structure/graph/all_features.pkl', 'rb') as f:
                self.all_features = pickle.load(f)


    def __len__(self):
        return len(self.x)

    def get_nnlqp(self, key):
        key = key.astype(np.int32) - 1
        graph_adj = self.all_graphs[key]       # scipy csr matrix, shape (n, n)
        features = self.all_features[key]      # shape (n, d)
        graph_adj = csr_matrix(graph_adj)  # 显式转换为 scipy csr matrix
        graph = from_scipy(graph_adj)
        graph = dgl.to_bidirected(graph)       # 如需无向图建模
        return graph, features
    
    def get_bench_201(self, key):
        graph, label = get_matrix_and_ops(key)
        graph, features = get_adjacency_and_features(graph, label)
        graph = dgl.from_scipy(csr_matrix(graph))
        graph = dgl.to_bidirected(graph)
        features = torch.tensor(features).long()
        return graph, features
    
    def __getitem__(self, idx):
        key, value = self.x[idx], self.y[idx]
        value = torch.tensor(value).float()

        if self.config.dataset == 'nasbench201':
            graph, features = self.get_bench_201(key)
        elif self.config.dataset == 'nnlqp':
            graph, features = self.get_nnlqp(key)
            features = torch.tensor(features, dtype=torch.float32)  # ✅ 添加这句！

        return graph, features, value

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        graph, features, y = zip(*batch)
        graph, features, y = dgl.batch(graph), default_collate(features), default_collate(y)
        return graph, features, y
    
    
class SeqDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.mode = mode
        self.x = x
        if config.dataset == 'nasbench201':
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
        else:
            with open('./datasets/nnlqp/unseen_structure/graph/all_tokens.pkl', 'rb') as f:
                self.x = pickle.load(f)
            with open('./datasets/nnlqp/unseen_structure/graph/all_stat_features.pkl', 'rb') as f:
                self.all_stat_features = pickle.load(f)
            

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