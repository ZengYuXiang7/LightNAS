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
from dgl import from_scipy
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
from models.my_feature import (
    laplacian_node_ids_from_adj,
    preprocess_binary_inout_and_spd,
)


def padding_2d(data, max_len, padding_value):
    # 1. pad 邻接矩阵到 [max_len, max_len]
    N = data.size(0)
    pad_n = max_len - N
    padded_data = F.pad(
        data,
        pad=(
            0,
            pad_n,
            0,
            pad_n,
        ),  # (left, right, top, bottom) -> 这里右和下各 pad pad_n
        value=padding_value,
    )
    return padded_data


def padding_1d(data, max_len, padding_value):
    # 1. pad 邻接矩阵到 [max_len, max_len]
    N = data.size(0)
    pad_n = max_len - N
    padded_data = F.pad(
        data,
        pad=(
            0,
            pad_n,
        ),
        value=padding_value,
    )
    return padded_data


def padding_feat_2d(x: torch.Tensor, max_len: int, padding_value: float = 0.0):
    """
    x: [N, D] -> [max_len, D]
    只在第一维(N)上 pad，不动特征维(D)
    """
    N, D = x.shape
    if N >= max_len:
        return x[:max_len, :]
    pad_n = max_len - N
    # F.pad: (left, right, top, bottom)
    # 对 [N, D]：H=N, W=D
    return F.pad(
        x,
        pad=(0, 0, 0, pad_n),  # 宽度(D)不扩，height(N)底部 pad_n
        value=padding_value,
    )


class NasBenchDataset(Dataset):
    def __init__(self, data, mode, config):
        self.config = config
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data[self.config.predict_target])

    def __getitem__(self, idx):
        if self.config.model == "ours":

            if self.config.dataset == "201_acc":
                features = self.data["features"][idx]
                y = self.data[self.config.predict_target][idx]

                key = self.data["key"][idx]
                arch_str = get_arch_str_from_arch_vector(key)  # 架构向量转字符串
                adj_matrix, features = info2mat(arch_str)

                features = torch.tensor(features, dtype=torch.long)
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
                # 获得拉普拉斯节点相对位置特征
                eigvec = laplacian_node_ids_from_adj(adj_matrix, self.config.lp_d_model)
                # 获得节点高阶特征
                indgree, outdegree, dij = preprocess_binary_inout_and_spd(
                    adj_matrix,
                    include_self_loops_in_degree=False,
                    directed_for_spd=True,  # 如需无向最短路可改成 False
                )

                # padding
                N, max_len = len(adj_matrix), 8
                key_padding_mask = torch.zeros(max_len, dtype=torch.bool)
                key_padding_mask[:N] = True

                # print(indgree.shape, outdegree.shape, dij.shape)
                
                y *= 0.01 # 2025年12月22日17:36:36
                return (
                    adj_matrix,
                    features,
                    eigvec,
                    indgree,
                    outdegree,
                    dij,
                    key_padding_mask,
                    y,
                )

            # elif self.config.dataset in ["101_acc", "nnlqp"]:
            elif self.config.dataset in ["101_acc"]:
                adj_matrix = self.data["adj_matrix"][idx]  # 直接取邻接矩阵
                features = self.data["features"][idx]  # 直接取操作序列
                features = torch.tensor(features, dtype=torch.long)
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

                eigvec = laplacian_node_ids_from_adj(adj_matrix, self.config.lp_d_model)
                indgree, outdegree, dij = preprocess_binary_inout_and_spd(
                    adj_matrix,
                    include_self_loops_in_degree=False,
                    directed_for_spd=True,  # 如需无向最短路可改成 False
                )
                y = self.data[self.config.predict_target][idx].reshape(-1)

                # padding
                N, max_len = len(adj_matrix), (
                    7 if self.config.dataset == "101_acc" else 245
                )
                adj_matrix = padding_2d(adj_matrix, max_len=max_len, padding_value=0)
                features = padding_1d(
                    features,
                    max_len=max_len,
                    padding_value=7 if self.config.dataset == "101_acc" else 33,
                )  # 7代表NUll操作 33 代表Null操作
                dij = padding_2d(
                    dij,
                    max_len=max_len,
                    padding_value=14 if self.config.dataset == "101_acc" else 244,
                )
                eigvec = padding_feat_2d(eigvec, max_len=max_len, padding_value=0)
                indgree = padding_1d(indgree, max_len=max_len, padding_value=0)
                outdegree = padding_1d(outdegree, max_len=max_len, padding_value=0)
                key_padding_mask = torch.zeros(max_len, dtype=torch.bool)
                key_padding_mask[:N] = True
                # print(adj_padded.shape, features_padded.shape, key_padding_mask.shape)
                # print(adj_matrix.shape, features.shape, eigvec.shape, indgree.shape, outdegree.shape, dij.shape, y.shape)
                # print(adj_padded.shape, features_padded.shape, key_padding_mask.shape, eigvec.shape, indgree.shape, outdegree.shape, dij.shape, y.shape)
                
                y *= 0.01 # 2025年12月22日17:36:36
                
                return (
                    adj_matrix,
                    features,
                    eigvec,
                    indgree,
                    outdegree,
                    dij,
                    key_padding_mask,
                    y,
                )
            elif self.config.dataset == 'nnlqp':
                adj_matrix = self.data["adj_matrix"][idx]  # 直接取邻接矩阵
                features = self.data["features"][idx]  # 直接取操作序列
                features = torch.tensor(features, dtype=torch.long)
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

                eigvec = laplacian_node_ids_from_adj(adj_matrix, self.config.lp_d_model)
                indgree, outdegree, dij = preprocess_binary_inout_and_spd(
                    adj_matrix,
                    include_self_loops_in_degree=False,
                    directed_for_spd=True,  # 如需无向最短路可改成 False
                )
                y = self.data[self.config.predict_target][idx].reshape(-1)
                
                # padding
                N, max_len = len(adj_matrix), (
                    7 if self.config.dataset == "101_acc" else 245
                )
                key_padding_mask = torch.zeros(max_len, dtype=torch.bool)
                key_padding_mask[:N] = True
                
                return (
                    adj_matrix,
                    features,
                    eigvec,
                    indgree,
                    outdegree,
                    dij,
                    key_padding_mask,
                    y,
                )

        elif self.config.model == "flops":
            flops = self.data["flops"][idx]
            y = self.data[self.config.predict_target][idx]
            return flops, y

        elif self.config.model == "flops-mac":
            flops = self.data["flops"][idx]
            params = self.data["params"][idx]
            y = self.data[self.config.predict_target][idx]
            features = np.array([flops, params], dtype=np.float32).reshape(-1)
            return features, y

        elif self.config.model == "brp-nas":
            adj_matrix = self.data["adj_matrix"][idx]
            features = self.data["features"][idx]

            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            features = torch.tensor(features, dtype=torch.long)

            features = F.one_hot(features, num_classes=6 if self.config.dataset == "101_acc" else 32).float()
            y = self.data[self.config.predict_target][idx]
            return adj_matrix, features, y

        elif self.config.model == "gat":
            adj_matrix = self.data["adj_matrix"][idx]
            features = self.data["features"][idx]
            y = self.data[self.config.predict_target][idx]

            graph = dgl.from_scipy(csr_matrix(adj_matrix))
            graph = dgl.to_bidirected(graph)
            features = torch.tensor(features, dtype=torch.long)
            features = F.one_hot(features, num_classes=6 if self.config.dataset == "101_acc" else 32).float()

            return graph, features, y

        elif self.config.model in {"lstm", "gru"}:
            features = self.data["features"][idx]
            features = torch.tensor(features, dtype=torch.float32)
            y = self.data[self.config.predict_target][idx]
            return features, y

        elif self.config.model == "narformer":
            key = self.data["key"][idx]
            if self.config.dataset == "201_acc":
                arch_str = get_arch_str_from_arch_vector(key)  # 架构向量转字符串
                adj_mat, ops_idx = info2mat(arch_str)  # 得到邻接矩阵与操作
                tokens = tokenizer(
                    ops_idx, adj_mat, 32, 32, 32, "nerf"
                )  # 得到token表示
            elif self.config.dataset == "101_acc":
                adj_mat = self.data["adj_matrix"][idx]  # 直接取邻接矩阵
                ops_idx = self.data["features"][idx]  # 直接取操作序列
                tokens = tokenizer(
                    ops_idx, adj_mat, 32, 32, 32, "nerf"
                )  # 得到token表示
                tokens, adj_mat = padding_for_batch1(tokens, adj_mat)

            # print(type(tokens), tokens.shape)
            y = self.data[self.config.predict_target][idx]
            return tokens, y

        elif self.config.model == "narformer2":
            if self.config.dataset == "201_acc":
                key = self.data["key"][idx]
                arch_str = get_arch_str_from_arch_vector(key)  # 架构向量转字符串
                adj_mat, ops_idx = info2mat(arch_str)  # 得到邻接矩阵与操作
                tokens = tokenizer2(
                    ops_idx, adj_mat, 32, 32, "nerf", "pe"
                )  # 得到token表示
            elif self.config.dataset == "101_acc":
                adj_mat = self.data["adj_matrix"][idx]  # 直接取邻接矩阵
                ops_idx = self.data["features"][idx]  # 直接取操作序列
                tokens = tokenizer2(
                    ops_idx, adj_mat, 32, 32, "nerf", "pe"
                )  # 得到token表示
                tokens, adj_mat = padding_for_batch2(tokens, adj_mat)

            adj_mat = torch.tensor(adj_mat, dtype=torch.float32)
            adj_mat = adj_mat + torch.t(adj_mat)

            num_vertices = torch.tensor([len(ops_idx)])

            y = self.data[self.config.predict_target][idx]
            return tokens, adj_mat, num_vertices, y

        elif self.config.model == "nnformer":
            if self.config.dataset == "201_acc":
                key = self.data["key"][idx]
                arch_str = get_arch_str_from_arch_vector(key)  # 架构向量转字符串
                adj_mat, ops_idx = info2mat(arch_str)  # 得到邻接矩阵与操作
            elif self.config.dataset in ["101_acc", 'nnlqp']:
                adj_mat = self.data["adj_matrix"][idx]  # 直接取邻接矩阵
                ops_idx = self.data["features"][idx]  # 直接取操作序列

            num_vertices = len(ops_idx)
            code, rel_pos, code_depth = tokenizer3(
                ops_idx, adj_mat, num_vertices, 96, "nape"
            )
            code, rel_pos = padding_for_batch3(code, rel_pos)
            y = self.data[self.config.predict_target][idx]

            # code = code.clone().detach()
            rel_pos = torch.tensor(rel_pos, dtype=torch.long)
            num_vertices = torch.tensor([len(ops_idx)])
            adj_mat = torch.tensor(adj_mat, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            return code, rel_pos, code_depth, adj_mat, y

        else:
            raise ValueError(f"Unsupported model type: {self.config.model}")

    def custom_collate_fn(self, batch):
        if self.config.model == "ours":
            (
                adj_matrix,
                features,
                eigvec,
                indgree,
                outdegree,
                dij,
                key_padding_mask,
                y,
            ) = zip(*batch)
            return (
                default_collate(adj_matrix).to(torch.float32),
                default_collate(features).to(torch.long),
                default_collate(eigvec).to(torch.float32),
                default_collate(indgree).to(torch.long),
                default_collate(outdegree).to(torch.long),
                default_collate(dij).to(torch.long),
                default_collate(key_padding_mask).to(torch.bool),
                default_collate(y).to(torch.float32),
            )
        elif self.config.model in {"flops", "flops-mac"}:
            features, y = zip(*batch)
            return default_collate(features).to(torch.float32), default_collate(y).to(
                torch.float32
            )
        elif self.config.model == "brp-nas":
            graph, features, y = zip(*batch)
            return (
                default_collate(graph).to(torch.float32),
                default_collate(features).to(torch.float32),
                default_collate(y).to(torch.float32),
            )
        elif self.config.model == "gat":
            graph, features, y = zip(*batch)
            return (
                dgl.batch(graph),
                default_collate(features).to(torch.long),
                default_collate(y).to(torch.float32),
            )
        elif self.config.model in {"lstm", "gru"}:
            features, y = zip(*batch)
            return default_collate(features).to(torch.float32), default_collate(y).to(
                torch.float32
            )
        elif self.config.model == "narformer":
            tokens, y = zip(*batch)
            return default_collate(tokens).to(torch.float32), default_collate(y).to(
                torch.float32
            )
        elif self.config.model == "narformer2":
            tokens, adj_mat, num_vertices, y = zip(*batch)
            return (
                default_collate(tokens).to(torch.float32),
                default_collate(adj_mat).to(torch.float32),
                default_collate(num_vertices).to(torch.long),
                default_collate(y).to(torch.float32),
            )
        elif self.config.model == "nnformer":
            code, rel_pos, code_depth, adj_mat, y = zip(*batch)
            return (
                default_collate(code).to(torch.float32),
                default_collate(rel_pos).to(torch.long),
                default_collate(code_depth).to(torch.float32),
                default_collate(adj_mat).to(torch.float32),
                default_collate(y).to(torch.float32),
            )

    def our_method_padding(self, adj_matrix, features, eigvec, indgree, outdegree, dij):
        N = adj_matrix.shape[0]  # 当前样本的节点数
        max_N = self.max_node  # 全局最大节点数（提前设定）

        # 1. adj_matrix: (N, N) → (max_N, max_N)，填充 0（无连接）
        adj_pad = torch.nn.functional.pad(
            adj_matrix,
            pad=(0, max_N - N, 0, max_N - N),  # 左右、上下各补多少
            value=0.0,
        )

        # 2. features: 若为 (N,) 先扩为 (N, 1)，再 pad 到 (max_N, F)，填充 -1（特殊占位符）
        if features.dim() == 1:
            features = features.unsqueeze(1)  # (N,) → (N, 1)
        F = features.shape[1]  # 特征维度
        feat_pad = torch.nn.functional.pad(
            features,
            pad=(0, 0, 0, max_N - N),  # 只在节点维度补（第二个维度不补）
            value=-1,  # 用 -1 标记 padding 特征（避免与真实特征冲突）
        )

        # 3. eigvec: (N, D) → (max_N, D)，填充 0
        D = self.config.lp_d_model
        eigvec_pad = torch.nn.functional.pad(
            eigvec, pad=(0, 0, 0, max_N - N), value=0.0  # D 维度不变，节点维度补 0
        )

        # 4. indgree: (N,) → (max_N,)，填充 0（无入度）
        indgree_pad = torch.nn.functional.pad(indgree, pad=(0, max_N - N), value=0.0)

        # 5. outdegree: (N,) → (max_N,)，填充 0（无出度）
        outdegree_pad = torch.nn.functional.pad(
            outdegree, pad=(0, max_N - N), value=0.0
        )

        # 6. dij: (N, N) → (max_N, max_N)，填充 inf（无路径）
        dij_pad = torch.nn.functional.pad(
            dij,
            pad=(0, max_N - N, 0, max_N - N),
            value=float("inf"),  # 用 inf 标记 padding 节点间的路径（与真实路径区分）
        )
        return adj_pad, feat_pad, eigvec_pad, indgree_pad, outdegree_pad, dij_pad
