# coding : utf-8
# Author : yuxiang Zeng

import torch
import dgl
from dgl.nn.pytorch import SAGEConv

from utils.config import get_config


class GraphSAGEConv(torch.nn.Module):
    def __init__(self, input_dim, rank, order, args):
        super(GraphSAGEConv, self).__init__()
        self.args = args
        self.rank = rank
        self.order = order
        self.layers = torch.nn.ModuleList([dgl.nn.pytorch.GraphConv(rank if i != 0 else input_dim, rank) for i in range(order)])
        self.dnn_embedding = torch.nn.Embedding(6, rank)
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(rank) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(order)])
        self.dropout = torch.nn.Dropout(0.10)
        self.pred_layers = torch.nn.Linear(rank, 1)
        print(self)

    def forward(self, graph, features):
        g, feats = graph, self.dnn_embedding(features).reshape(features.shape[0] * 9, -1)
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats)
            print(feats.shape)
            feats = norm(feats)
            feats = act(feats)
            feats = self.dropout(feats)
        batch_sizes = torch.as_tensor(g.batch_num_nodes()).to(self.args.device)  # 每个图的节点数
        first_nodes_idx = torch.cumsum(torch.cat((torch.tensor([0]).to(self.args.device), batch_sizes[:-1])), dim=0)  # 使用torch.cat来连接Tensor
        # 获取每个图的首个节点的特征
        first_node_features = feats[first_nodes_idx]
        y = self.pred_layers(first_node_features)
        return y


if __name__ == '__main__':
    # Build a random graph
    args = get_config()
    num_nodes, num_edges = 100, 200
    src_nodes = torch.randint(0, num_nodes, (num_edges,))
    dst_nodes = torch.randint(0, num_nodes, (num_edges,))
    print(src_nodes.shape, dst_nodes.shape)

    graph = dgl.graph((src_nodes, dst_nodes))
    graph = dgl.add_self_loop(graph)

    # Demo test
    bs = 32
    features = torch.randn(num_nodes, 64)
    graph_gcn = GraphSAGEConv(64, 128, 2, args)
    # print(graph_gcn)
    embeds = graph_gcn(graph, features)
    print(embeds.shape)
