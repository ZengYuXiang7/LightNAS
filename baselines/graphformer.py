import torch
import dgl
from dgl.nn.pytorch import GraphConv
from torch import nn

class Graphormer(torch.nn.Module):
    def __init__(self, input_dim, config):
        super(Graphormer, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.order = config.order

        # 定义图卷积层和其他模块
        self.layers = torch.nn.ModuleList([GraphConv(input_dim if i == 0 else self.d_model, self.d_model) for i in range(self.order)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(self.d_model) for _ in range(self.order)])
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.pred_layer = torch.nn.Linear(self.d_model, 1)

    def forward(self, graph, features):
        g = graph
        feats = features

        # 应用层和归一化
        for layer, norm in zip(self.layers, self.norms):
            feats = layer(g, feats)
            feats = norm(feats)
            feats = self.act(feats)
            feats = self.dropout(feats)

        # 输出层
        output = self.pred_layer(feats)
        return output

# 示例配置
class Config:
    def __init__(self):
        self.d_model = 128
        self.order = 4
        self.device = 'cuda'

config = Config()

# 用于测试的虚拟图和特征
graph = dgl.rand_graph(100, 500)  # 100个节点，500条边
features = torch.randn(100, config.d_model)

# 创建并测试Graphormer
model = Graphormer(input_dim=config.d_model, config=config)
output = model(graph, features)
print(output.shape)

