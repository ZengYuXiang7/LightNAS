# coding : utf-8
# Author : yuxiang Zeng
# 根据场景需要来改这里的input形状
from torch.utils.data import Dataset
import numpy as np
import os
from baselines.narformer import padding_for_batch1, tokenizer
from baselines.narformer2 import padding_for_batch2, tokenizer2
from baselines.nnformer import padding_for_batch3, tokenizer3
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
            features = torch.tensor(features, dtype=torch.long).unsqueeze(-1)
            
            # graph = dgl.from_scipy(csr_matrix(adj_matrix))
            # graph = dgl.to_bidirected(graph)
            graph = adj_matrix
            
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
            if self.config.dataset == '201_acc':
                arch_str = get_arch_str_from_arch_vector(key)             # 架构向量转字符串
                adj_mat, ops_idx = info2mat(arch_str)                     # 得到邻接矩阵与操作
                tokens = tokenizer(ops_idx, adj_mat, 32, 32, 32, 'nerf')  # 得到token表示
            elif self.config.dataset == '101_acc':
                adj_mat = self.data['adj_matrix'][idx]             # 直接取邻接矩阵
                ops_idx = self.data['features'][idx]             # 直接取操作序列
                tokens = tokenizer(ops_idx, adj_mat, 32, 32, 32, 'nerf')  # 得到token表示
                tokens, adj_mat = padding_for_batch1(tokens, adj_mat)
            
            # print(type(tokens), tokens.shape)
            y      = self.data[self.config.predict_target][idx]
            return tokens, y
        
        elif self.config.model == 'narformer2':
            if self.config.dataset == '201_acc':
                key = self.data['key'][idx]
                arch_str = get_arch_str_from_arch_vector(key)             # 架构向量转字符串
                adj_mat, ops_idx = info2mat(arch_str)                     # 得到邻接矩阵与操作
                tokens = tokenizer2(ops_idx, adj_mat, 32, 32, 'nerf', 'pe')  # 得到token表示
            elif self.config.dataset == '101_acc':
                adj_mat = self.data['adj_matrix'][idx]             # 直接取邻接矩阵
                ops_idx = self.data['features'][idx]             # 直接取操作序列
                tokens = tokenizer2(ops_idx, adj_mat, 32, 32, 'nerf', 'pe')  # 得到token表示
                tokens, adj_mat = padding_for_batch2(tokens, adj_mat)
            
            adj_mat = torch.tensor(adj_mat, dtype=torch.float32)
            adj_mat = adj_mat + torch.t(adj_mat)
            
            num_vertices = torch.tensor([len(ops_idx)])
            
            y      = self.data[self.config.predict_target][idx]
            return tokens, adj_mat, num_vertices, y
        
        elif self.config.model == 'nnformer':
            if self.config.dataset == '201_acc':
                key = self.data['key'][idx]
                arch_str = get_arch_str_from_arch_vector(key)             # 架构向量转字符串
                adj_mat, ops_idx = info2mat(arch_str)                     # 得到邻接矩阵与操作
            elif self.config.dataset == '101_acc':
                adj_mat = self.data['adj_matrix'][idx]             # 直接取邻接矩阵
                ops_idx = self.data['features'][idx]             # 直接取操作序列
                
                
            num_vertices = len(ops_idx)
            code, rel_pos, code_depth = tokenizer3(ops_idx, adj_mat, num_vertices, 96, 'nape')
            code, rel_pos = padding_for_batch3(code, rel_pos)
            y      = self.data[self.config.predict_target][idx]
            
            # code = code.clone().detach()
            rel_pos = torch.tensor(rel_pos, dtype=torch.long)
            num_vertices = torch.tensor([len(ops_idx)])
            adj_mat = torch.tensor(adj_mat, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            
            return code, rel_pos, code_depth, adj_mat, y
        
        else: 
            raise ValueError(f"Unsupported model type: {self.config.model}")
        
        
    def custom_collate_fn(self, batch, config):
        if self.config.model == 'ours':
            graph, features, y = zip(*batch)
            return default_collate(graph).to(torch.float32), default_collate(features).to(torch.long), default_collate(y).to(torch.float32)
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
        elif self.config.model == 'narformer2':
            tokens, adj_mat, num_vertices, y = zip(*batch)
            return default_collate(tokens).to(torch.float32), default_collate(adj_mat).to(torch.float32), default_collate(num_vertices).to(torch.long), default_collate(y).to(torch.float32)
        elif self.config.model == 'nnformer':
            code, rel_pos, code_depth, adj_mat, y = zip(*batch)
            return default_collate(code).to(torch.float32), default_collate(rel_pos).to(torch.long), default_collate(code_depth).to(torch.float32), default_collate(adj_mat).to(torch.float32), default_collate(y).to(torch.float32)
        

