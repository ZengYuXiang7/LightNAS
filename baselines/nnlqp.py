import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn


def init_tensor(tensor, init_type, nonlinearity):
    if tensor is None or init_type is None:
        return
    if init_type == 'thomas':
        size = tensor.size(-1)
        stdv = 1. / math.sqrt(size)
        nn.init.uniform_(tensor, -stdv, stdv)
    elif init_type == 'kaiming_normal_in':
        nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_normal_out':
        nn.init.kaiming_normal_(tensor, mode='fan_out', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_uniform_in':
        nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_uniform_out':
        nn.init.kaiming_uniform_(tensor, mode='fan_out', nonlinearity=nonlinearity)
    elif init_type == 'orthogonal':
        nn.init.orthogonal_(tensor, gain=nn.init.calculate_gain(nonlinearity))
    else:
        raise ValueError(f'Unknown initialization type: {init_type}')


class PredictFC(nn.Module):
    def __init__(self, input_feature, fc_hidden):
        super().__init__()
        self.fc_1 = nn.Linear(input_feature, fc_hidden)
        self.fc_2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc_relu_1 = nn.ReLU()
        self.fc_relu_2 = nn.ReLU()
        self.fc_drop_1 = nn.Dropout(0.05)
        self.fc_drop_2 = nn.Dropout(0.05)
        self.predictor = nn.Linear(fc_hidden, 1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_relu_1(x)
        x = self.fc_drop_1(x)
        x = self.fc_2(x)
        x = self.fc_relu_2(x)
        x = self.fc_drop_2(x)
        x = self.predictor(x)
        return x


def _make_gnn_layer(name: str, in_dim: int, out_dim: int):
    """最小映射：只实现 PyG 里你常用的 SAGEConv / GraphConv 的等价层。"""
    if name == "SAGEConv":
        # PyG 里 normalize=True 的 SAGEConv 在 DGL 无直接参数；选常用 'mean' 聚合
        return dglnn.SAGEConv(in_feats=in_dim, out_feats=out_dim, aggregator_type='mean')
    if name == "GraphConv":
        # 与 PyG GraphConv(normalize=True) 对齐，双向归一化
        return dglnn.GraphConv(in_feat=in_dim, out_feat=out_dim, norm='both', weight=True, bias=True)
    # 如需更多 GNN 层，按需扩展
    raise ValueError(f"Unsupported gnn_layer='{name}'. Use 'SAGEConv' or 'GraphConv'.")


def _graph_readout(g: dgl.DGLGraph, feat_key: str, reduce_func: str):
    reduce_func = reduce_func.lower()
    if reduce_func == "sum":
        return dgl.readout.sum_nodes(g, feat_key)
    if reduce_func == "mean":
        return dgl.readout.mean_nodes(g, feat_key)
    if reduce_func == "max":
        return dgl.readout.max_nodes(g, feat_key)
    if reduce_func == "min":
        return dgl.readout.min_nodes(g, feat_key)
    # DGL 没有 product 读出；需要的话可以改成 log-域求和再 exp 的近似实现
    raise ValueError(f"reduce_func='{reduce_func}' is not supported in DGL readout (no 'mul').")


class NNLQP(nn.Module):
    def __init__(
        self,
        num_node_features=44,
        gnn_layer="SAGEConv",
        gnn_hidden=512,
        fc_hidden=512,
        reduce_func="sum",
        multi_plt=None,
        norm_sf=False,
    ):
        super().__init__()
        self.reduce_func = reduce_func
        self.num_node_features = num_node_features
        self.multi_plt = multi_plt or {}
        self.norm_sf = norm_sf
        self.gnn_layer_name = gnn_layer

        # GNN 两层
        self.graph_conv_1 = _make_gnn_layer(gnn_layer, num_node_features, gnn_hidden)
        self.graph_conv_2 = _make_gnn_layer(gnn_layer, gnn_hidden, gnn_hidden)
        self.gnn_drop_1 = nn.Dropout(0.05)
        self.gnn_drop_2 = nn.Dropout(0.05)
        self.gnn_relu1 = nn.ReLU()
        self.gnn_relu2 = nn.ReLU()

        out_dim = 1 if len(self.multi_plt) <= 1 else len(self.multi_plt)

        if out_dim > 1:
            # 与原版一致：多头时固定拼接 gnn_hidden + 4（不做 norm_sf）
            heads = [PredictFC(gnn_hidden + 4, fc_hidden) for _ in range(out_dim)]
            self.heads = nn.ModuleList(heads)
            self._use_heads = True
        else:
            self._use_heads = False
            if self.norm_sf:
                self.norm_sf_linear = nn.Linear(4, gnn_hidden)
                self.norm_sf_drop = nn.Dropout(0.05)
                self.norm_sf_relu = nn.ReLU()
                sf_hidden = gnn_hidden
            else:
                sf_hidden = 4
            self.fc_1 = nn.Linear(gnn_hidden + sf_hidden, fc_hidden)
            self.fc_2 = nn.Linear(fc_hidden, fc_hidden)
            self.fc_drop_1 = nn.Dropout(0.05)
            self.fc_drop_2 = nn.Dropout(0.05)
            self.fc_relu1 = nn.ReLU()
            self.fc_relu2 = nn.ReLU()
            self.predictor = nn.Linear(fc_hidden, out_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_tensor(m.weight, "thomas", "relu")
                init_tensor(m.bias, "thomas", "relu")
            # DGL 的图卷积层内部参数保持默认初始化即可

    def forward(self, g: dgl.DGLGraph, static_feature: torch.Tensor):
        """
        g: batched DGLGraph, 节点特征在 g.ndata['x']，shape [N, num_node_features]
        static_feature: shape [B, 4]（或在 norm_sf=True 时仍为 [B,4]，内部再升维）
        """
        x = g.ndata['x']  # [N, F]

        x = self.graph_conv_1(g, x)   # [N, gnn_hidden]
        x = self.gnn_relu1(x)
        x = self.gnn_drop_1(x)

        x = self.graph_conv_2(g, x)   # [N, gnn_hidden]
        x = self.gnn_relu2(x)
        x = self.gnn_drop_2(x)

        g.ndata['h'] = x
        gnn_feat = _graph_readout(g, 'h', self.reduce_func)  # [B, gnn_hidden]

        if self._use_heads:
            # 多头时与原实现一致：直接拼接未归一化的 4 维静态特征
            x_all = torch.cat([gnn_feat, static_feature], dim=1)  # [B, gnn_hidden+4]
            outs = [head(x_all) for head in self.heads]           # list of [B,1]
            x_out = torch.cat(outs, dim=1)                        # [B, out_dim]
        else:
            if self.norm_sf:
                sf = self.norm_sf_linear(static_feature)
                sf = self.norm_sf_drop(sf)
                sf = self.norm_sf_relu(sf)
            else:
                sf = static_feature
            x_all = torch.cat([gnn_feat, sf], dim=1)              # [B, gnn_hidden+sf_hidden]
            x_all = self.fc_1(x_all)
            x_all = self.fc_relu1(x_all)
            x_all = self.fc_drop_1(x_all)
            x_all = self.fc_2(x_all)
            x_all = self.fc_relu2(x_all)
            feat = self.fc_drop_2(x_all)
            x_out = self.predictor(feat)                          # [B, out_dim]

        pred = -F.logsigmoid(x_out)  # (0, +inf)
        return pred
    
    
    
# 构造一个小 batch（示例）
import dgl
import torch

# 2 个子图，各 3 个节点
edges1 = (torch.tensor([0,1,2]), torch.tensor([1,2,0]))
edges2 = (torch.tensor([0,1]),   torch.tensor([1,2]))
g1 = dgl.graph(edges1)
g2 = dgl.graph(edges2)

num_node_features = 44
g1.ndata['x'] = torch.randn(g1.num_nodes(), num_node_features)
g2.ndata['x'] = torch.randn(g2.num_nodes(), num_node_features)

bg = dgl.batch([g1, g2])                 # 批量图
static_feature = torch.randn(2, 4)       # 对应 batch=2 的每图 4 维静态特征

model = NetDGL(num_node_features=44, gnn_layer="SAGEConv",
               gnn_hidden=128, fc_hidden=128,
               reduce_func="sum", multi_plt={}, norm_sf=False)

out = model(bg, static_feature)          # [2,1]
print(out.shape, out[:2])