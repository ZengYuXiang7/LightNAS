# coding : utf-8
# Author : yuxiang Zeng

import torch


class ARNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ARNN, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, adj_matrix, x):
        # x: 节点特征矩阵，形状为 (batch_size, num_nodes, feature_size)
        # adj_matrix: 批处理的图的邻接矩阵，形状为 (batch_size, num_nodes, num_nodes)
        batch_size, num_nodes, _ = x.shape
        x = x.to(torch.float32)
        adj_matrix = adj_matrix.to(torch.float32)

        updated_features_batch = []

        # 处理每个样本
        for b in range(batch_size):
            updated_features = []
            for i in range(num_nodes):  # 遍历所有节点
                neighbors_indices = (adj_matrix[b, i] > 0).nonzero(as_tuple=False).view(-1)
                node_features = x[b, i, :]  # 当前节点特征
                neighbor_features = [node_features.unsqueeze(0)]  # 包括节点自身，增加一个维度以匹配

                for neighbor_index in neighbors_indices:
                    neighbor_features.append(x[b, neighbor_index, :].unsqueeze(0))  # 添加邻居节点特征

                # 计算平均特征向量
                neighbor_features = torch.cat(neighbor_features, dim=0)
                avg_feature = torch.mean(neighbor_features, dim=0, keepdim=True)
                updated_features.append(avg_feature.squeeze(0))  # 移除多余的维度

            # 将更新后的特征向量堆叠为一个新的特征矩阵
            updated_features_batch.append(torch.stack(updated_features, dim=0))

        updated_features_batch = torch.stack(updated_features_batch, dim=0)
        updated_features_batch = updated_features_batch.float()

        # 经过RNN
        out, (hn, cn) = self.rnn(updated_features_batch)

        # 这里是处理双向LSTM的逻辑，根据你的LSTM配置可能需要调整
        hn_fwd = hn[-2, :, :]  # 前向的最后隐藏状态
        hn_bwd = hn[-1, :, :]  # 后向的最后隐藏状态
        hn_combined = torch.cat((hn_fwd, hn_bwd), dim=1)  # 形状: (batch_size, hidden_dim * 2)

        return hn_combined
