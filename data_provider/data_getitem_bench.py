# coding : utf-8
# Author : yuxiang Zeng
# 根据场景需要来改这里的input形状
from torch.utils.data import Dataset
import numpy as np
import os
from baselines.narformer import tokenizer
from data_process.create_latency import *
from scipy.sparse import csr_matrix
import dgl
from tqdm import *
import pickle 
from dgl import from_scipy
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

class NasBenchDataset(Dataset):
    def __init__(self, data, mode, config):
        self.config = config
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data[self.config.predict_target])

    def __getitem__(self, idx):
        if self.config.model == 'ours':
        
            adj_matrix = self.data['adj_matrix'][idx]
            features   = self.data['features'][idx]
            y          = self.data[self.config.predict_target][idx]
            
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            features = torch.tensor(features, dtype=torch.long)
            
            graph = dgl.from_scipy(csr_matrix(adj_matrix))
            graph = dgl.to_bidirected(graph)
            
            return graph, features, y
        
        elif self.config.model == "flops":
            flops = self.data['flops'][idx]
            y     = self.data[self.config.predict_target][idx]
            return flops, y
        elif self.config.model == "flops-mac":
            flops = self.data['flops'][idx]
            params= self.data['params'][idx]
            y     = self.data[self.config.predict_target][idx]
            features = np.array([flops, params], dtype=np.float32).reshape(-1)
            return features, y
        
        elif self.config.model == 'brp-nas':
            adj_matrix = self.data['adj_matrix'][idx]
            features   = self.data['features'][idx]
            
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            features = torch.tensor(features, dtype=torch.long)
            
            features = F.one_hot(features, num_classes=6).float()
            y          = self.data[self.config.predict_target][idx]
            return adj_matrix, features, y
        
        elif self.config.model == 'gat':
            adj_matrix = self.data['adj_matrix'][idx]
            features   = self.data['features'][idx]
            y          = self.data[self.config.predict_target][idx]
            
            graph = dgl.from_scipy(csr_matrix(adj_matrix))
            graph = dgl.to_bidirected(graph)
            features = torch.tensor(features, dtype=torch.long)
            features = F.one_hot(features, num_classes=6).float()
            
            return graph, features, y
        
        elif self.config.model in {'lstm', 'gru'}:
            features = self.data['features'][idx]
            features = torch.tensor(features, dtype=torch.float32)
            y        = self.data[self.config.predict_target][idx]
            return features, y
        
        elif self.config.model == 'narformer':
            key = self.data['key'][idx]
            arch_str = get_arch_str_from_arch_vector(key)             # 架构向量转字符串
            adj_mat, ops_idx = info2mat(arch_str)                     # 得到邻接矩阵与操作
            tokens = tokenizer(ops_idx, adj_mat, 32, 32, 32, 'nerf')  # 得到token表示
            y      = self.data[self.config.predict_target][idx]
            return tokens, y
        else: 
            raise ValueError(f"Unsupported model type: {self.config.model}")
        
        
    def custom_collate_fn(self, batch, config):
        if self.config.model == 'ours':
            graph, features, y = zip(*batch)
            return dgl.batch(graph), default_collate(features).to(torch.long), default_collate(y).to(torch.float32)
        elif self.config.model in {"flops", "flops-mac"}:
            features, y = zip(*batch)
            return default_collate(features).to(torch.float32), default_collate(y).to(torch.float32)
        elif self.config.model == 'brp-nas':
            graph, features, y = zip(*batch)
            return default_collate(graph).to(torch.float32), default_collate(features).to(torch.float32), default_collate(y).to(torch.float32)
        elif self.config.model == 'gat':
            graph, features, y = zip(*batch)
            return dgl.batch(graph), default_collate(features).to(torch.long), default_collate(y).to(torch.float32)
        elif self.config.model in {'lstm', 'gru'}:
            features, y = zip(*batch)
            return default_collate(features).to(torch.float32), default_collate(y).to(torch.float32)
        elif self.config.model == 'narformer':
            tokens, y = zip(*batch)
            return default_collate(tokens).to(torch.float32), default_collate(y).to(torch.float32)


        

