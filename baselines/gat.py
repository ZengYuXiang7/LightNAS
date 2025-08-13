# coding : utf-8
# Author : yuxiang Zeng

import torch
import dgl
from dgl.nn.pytorch import SAGEConv


class GAT(torch.nn.Module):
    def __init__(self, input_dim, config):
        super(GAT, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.order = config.order
        self.layers = torch.nn.ModuleList([dgl.nn.pytorch.GraphConv(self.d_model if i != 0 else input_dim, self.d_model) for i in range(self.order)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(self.d_model) for _ in range(self.order)])
        self.acts = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(self.order)])
        self.dropout = torch.nn.Dropout(0.10)
        self.pred_layers = torch.nn.Linear(self.d_model, 1)

    def forward(self, graph, features):
        g, feats = graph, features.reshape(features.shape[0] * 9, -1)
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats)
            feats = norm(feats)
            feats = act(feats)
            feats = self.dropout(feats)
        batch_sizes = torch.as_tensor(g.batch_num_nodes()).to(self.config.device)  # 每个图的节点数
        first_nodes_idx = torch.cumsum(torch.cat((torch.tensor([0]).to(self.config.device), batch_sizes[:-1])), dim=0)  # 使用torch.cat来连接Tensor
        first_node_features = feats[first_nodes_idx]
        y = self.pred_layers(first_node_features)
        return y


if __name__ == '__main__':
    # Build a random graph
    config = get_config()
    num_nodes, num_edges = 100, 200
    src_nodes = torch.randint(0, num_nodes, (num_edges,))
    dst_nodes = torch.randint(0, num_nodes, (num_edges,))
    print(src_nodes.shape, dst_nodes.shape)

    graph = dgl.graph((src_nodes, dst_nodes))
    graph = dgl.add_self_loop(graph)

    # Demo test
    bs = 32
    features = torch.randn(num_nodes, 64)
    graph_gcn = GraphSAGEConv(64, 128, 2, config)
    # print(graph_gcn)
    embeds = graph_gcn(graph, features)
    print(embeds.shape)
