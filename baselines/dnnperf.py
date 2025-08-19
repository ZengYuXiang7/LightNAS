import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
from scipy.sparse import csr_matrix

from models.latency_data import get_matrix_and_ops, get_adjacency_and_features


# ANEENodeUpdate类，用于更新节点特征
class ANEENodeUpdate(torch.nn.Module):
    def __init__(self, in_feats, out_feats, alpha=0.01):
        super(ANEENodeUpdate, self).__init__()
        self.W_u = torch.nn.Linear(in_feats, out_feats, bias=False)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=alpha)

    def forward(self, h):
        h_new = self.leaky_relu(self.W_u(h))
        return h_new


# ANEEEdgeUpdate类，用于更新边特征
class ANEEEdgeUpdate(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(ANEEEdgeUpdate, self).__init__()
        self.W_e = torch.nn.Linear(in_feats, out_feats, bias=False)  # 线性变换，没有偏置项
        self.a = torch.nn.Parameter(torch.Tensor(2 * out_feats, 1))  # 定义参数a
        torch.nn.init.xavier_uniform_(self.a)  # 使用Xavier初始化a参数

    def forward(self, h_s, h_d, e):
        # 前向传播，更新边特征
        combined = torch.cat([h_s, h_d], dim=-1)  # 将源节点和目标节点特征拼接在一起
        score = combined @ self.a  # 计算注意力分数
        edge_update = torch.sigmoid(score) * self.W_e(e)  # 使用sigmoid函数加权边特征，并线性变换
        return edge_update


# ANEENodeFinalUpdate类，用于基于边特征更新最终的节点特征
class ANEENodeFinalUpdate(torch.nn.Module):
    def __init__(self, in_feats, out_feats, alpha=0.01):
        # 初始化函数，定义输入输出特征维度和LeakyReLU的负斜率alpha
        super(ANEENodeFinalUpdate, self).__init__()
        self.W_m = torch.nn.Linear(out_feats, 1, bias=False)  # 线性变换，没有偏置项
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=alpha)  # 使用LeakyReLU激活函数

    def forward(self, edges):
        # 前向传播，基于边特征和节点特征更新节点
        e = edges.data['e']  # 获取边特征
        h_prime = edges.src['h_prime']  # 获取源节点特征
        softmax_scores = F.softmax(self.W_m(e), dim=0)  # 对边特征的线性变换结果进行softmax操作
        h_new = softmax_scores * h_prime
        # h_new = self.leaky_relu(torch.sum(h_new, dim=1))  # 使用加权节点特征的和作为新的节点特征
        h_new = self.leaky_relu(h_new)  # 使用加权节点特征的和作为新的节点特征
        return {'h_new': h_new}  # 返回更新后的节点特征


# ANEEGlobalAggregation类，用于进行全局图级特征聚合
class ANEEGlobalAggregation(torch.nn.Module):
    def __init__(self, in_feats):
        # 初始化函数，定义输入特征维度，并使用多层感知机（MLP）进行全局聚合
        super(ANEEGlobalAggregation, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, 512),  # 输入特征到512维
            torch.nn.ReLU(),  # ReLU激活
            torch.nn.Linear(512, 128),  # 512维到128维
            torch.nn.ReLU(),  # ReLU激活
            torch.nn.Linear(128, 16),  # 128维到16维
            torch.nn.ReLU(),  # ReLU激活
            torch.nn.Linear(16, 1),  # 16维到1维
        )

    def forward(self, g):
        # 前向传播，执行全局节点特征的聚合操作
        with g.local_scope():
            print(g)
            print(g.ndata['h_new'].shape)
            hg = dgl.sum_nodes(g, 'h_new')  # 计算每个图的所有节点特征之和
            print(hg.shape)
            exit()
            return self.mlp(hg)  # 使用MLP进行全局聚合，并返回结果


# DNNPerf类，整体模型架构，包含节点、边更新和全局聚合操作
class DNNPerf(torch.nn.Module):
    def __init__(self, in_node_feats, node_hidden_feats, in_edge_feats, edge_hidden_feats, args):
        # 初始化函数，定义节点和边的输入输出特征维度以及模型的参数
        super(DNNPerf, self).__init__()
        self.args = args  # 存储模型参数
        self.node_update = ANEENodeUpdate(in_node_feats, node_hidden_feats)  # 初始化节点更新模块
        self.edge_update = ANEEEdgeUpdate(in_edge_feats, edge_hidden_feats)  # 初始化边更新模块
        self.final_node_update = ANEENodeFinalUpdate(edge_hidden_feats, node_hidden_feats)  # 初始化最终节点更新模块
        self.global_agg = ANEEGlobalAggregation(node_hidden_feats)  # 初始化全局聚合模块

    def forward(self, g, _):
        # 前向传播，执行图的节点和边的特征更新操作
        with g.local_scope():
            g.ndata['h_prime'] = self.node_update(g.ndata['h'])  # 更新节点特征
            g.apply_edges(lambda edges: {'e': self.edge_update(edges.src['h_prime'], edges.dst['h_prime'], edges.data['e'])})  # 更新边特征
            g.update_all(self.final_node_update, fn.sum('h_new', 'h_new'))  # 更新所有节点特征并进行聚合
            return self.global_agg(g)  # 返回全局聚合的结果


def calculate_flops_and_tensor_sizes(layer, input_shape, sz=4):
    N, C_i, H_i, W_i = input_shape

    if isinstance(layer, torch.nn.Conv2d):
        C_o = layer.out_channels
        k_s = layer.kernel_size
        pad = layer.padding
        sd = layer.stride
        dil = layer.dilation
    elif isinstance(layer, torch.nn.MaxPool2d):
        C_o = C_i
        k_s = layer.kernel_size
        pad = layer.padding
        sd = layer.stride
        dil = (1, 1)
    elif isinstance(layer, POOLING):
        C_o = C_i
        k_s = layer.op.kernel_size
        pad = layer.op.padding
        sd = layer.op.stride
        dil = (1, 1)
    elif isinstance(layer, ReLUConvBN):
        C_o = layer.op[1].out_channels  # Conv2d 层
        k_s = layer.op[1].kernel_size
        pad = layer.op[1].padding
        sd = layer.op[1].stride
        dil = layer.op[1].dilation
    elif layer == 'input' or layer == 'output':
        C_o = C_i
        k_s = (1, 1)
        pad = (0, 0)
        sd = (1, 1)
        dil = (1, 1)

    if isinstance(k_s, int):
        k_s = (k_s, k_s)
    if isinstance(pad, int):
        pad = (pad, pad)
    if isinstance(sd, int):
        sd = (sd, sd)
    if isinstance(dil, int):
        dil = (dil, dil)

    def calculate_output_shape(H_i, W_i, pad, sd, dil, k_s):
        H_o = (H_i + 2 * pad[0] - dil[0] * (k_s[0] - 1) - 1) // sd[0] + 1
        W_o = (W_i + 2 * pad[1] - dil[1] * (k_s[1] - 1) - 1) // sd[1] + 1
        return H_o, W_o

    H_o, W_o = calculate_output_shape(H_i, W_i, pad, sd, dil, k_s)

    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, ReLUConvBN):
        FLOPs = 2 * C_i * k_s[0] * k_s[1] * H_o * W_o * C_o * N
        IT = sz * N * C_i * H_i * W_i
        OT = sz * N * C_o * H_o * W_o
        WT = sz * (C_i * k_s[0] * k_s[1] * C_o)
    elif isinstance(layer, torch.nn.MaxPool2d) or isinstance(layer, POOLING):
        FLOPs = C_i * H_o * W_o * N
        IT = sz * N * C_i * H_i * W_i
        OT = sz * N * C_o * H_o * W_o
        WT = 0
    elif layer == 'input' or layer == 'output':
        FLOPs = 0
        IT = sz * N * C_i * H_i * W_i
        OT = IT
        WT = 0

    return FLOPs, IT, OT, WT, (N, C_o, H_o, W_o)


def generate_node_vector(layer_type, input_shape, key, args):
    if isinstance(layer_type, (ReLUConvBN, POOLING)) or layer_type in ['input', 'output']:
        FLOPs, IT, OT, WT, output_shape = calculate_flops_and_tensor_sizes(layer_type, input_shape)
    else:
        FLOPs, IT, OT, WT, output_shape = 0, 0, 0, 0, input_shape

    h_t = torch.tensor([key], dtype=torch.float)
    h_p = torch.tensor([3], dtype=torch.float)
    h_d = torch.tensor([IT, OT, WT], dtype=torch.float)
    h_c = torch.tensor([FLOPs], dtype=torch.float)
    h_r = torch.tensor([16 if args.dataset == 'cpu' else 11], dtype=torch.float)

    h = torch.cat([h_t, h_p, h_d, h_c, h_r])
    return h, output_shape

"""
    Operation idx 
    2-------nor_conv_1x1 : POOLING(3, 1, 1)
    3-------nor_conv_3x3 : ReLUConvBN(16, 16, 1, 1, 0), 
    4-------avg_pool_3x3 : ReLUConvBN(16, 16, 3, 1, 1)
"""


class POOLING(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(POOLING, self).__init__()
        self.op = torch.nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return self.op(x)


class ReLUConvBN(torch.nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ReLUConvBN, self).__init__()
        self.op = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            torch.nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.op(x)


def generate_op_features(op_idx, args):
    op_model = [
        ReLUConvBN(16, 16, 1, 1, 0),
        ReLUConvBN(16, 16, 3, 1, 0),
        POOLING(3, 1, 1),
        'input',
        'output',
        'None'
    ]
    op_features = []
    for idx in op_idx:
        h, output_shape = generate_node_vector(op_model[idx], (1, 3, 32, 32), idx, args)
        # print(f"Layer Type: {op_model[idx].__class__.__name__ if isinstance(op_model[idx], torch.torch.nn.Module) else op_model[idx]}")
        # print(f"Final Node Feature Vector h: {h}\n")
        op_features.append(h)
    op_features = torch.stack(op_features, dim=0)
    return op_features


# 创建样例图的函数
def create_sample_graph(key, graph, args):
    # u = torch.tensor([0, 1, 2])
    # v = torch.tensor([1, 2, 3])
    # g = dgl.graph((u, v))
    g = graph

    # 创建节点特征和边特征
    # node_features = torch.rand(4, 5)
    node_features = generate_op_features(key, args)
    num_edges = g.num_edges()
    # DNNPerf 带宽内容
    variable_value = 85 if args.dataset == 'cpu' else 484
    edge_features = torch.zeros(num_edges, 2)
    # 设置每个边的第一个值为0，第二个值为 variable_value
    edge_features[:, 1] = variable_value

    def normalize_features(features):
        """
        对特征矩阵的每一列进行归一化。
        参数:
        - features: torch.Tensor, 大小为 [num_samples, num_features]

        返回:
        - normalized_features: torch.Tensor, 归一化后的特征矩阵，大小同 features
        """
        min_vals, _ = features.min(dim=1, keepdim=True)  # 计算每列的最小值
        max_vals, _ = features.max(dim=1, keepdim=True)  # 计算每列的最大值

        normalized_features = (features - min_vals) / (max_vals - min_vals + 1e-8)  # 归一化处理
        return normalized_features

    # 归一化节点特征和边特征
    # print(node_features)
    g.ndata['h'] = normalize_features(node_features)
    # print(g.ndata['h'])
    g.edata['e'] = normalize_features(edge_features)
    # print(node_features)
    return g


def main():
    """
        0 con1 1 con3 2 max3 3 input 4 output 5 None
    """
    keys = [
        [5, 2, 3, 1, 2, 2],
        [2, 4, 1, 0, 3, 2]
    ]

    model = DNNPerf(in_node_feats=7, node_hidden_feats=32, in_edge_feats=2, edge_hidden_feats=32, args=1)

    idx = 0
    # 创建多个样例图，模拟 batch
    inputs = keys[idx]
    graph, label = get_matrix_and_ops(inputs)
    graph, features = get_adjacency_and_features(graph, label)
    graph = dgl.to_bidirected(dgl.from_scipy(csr_matrix(graph)))
    key = torch.argmax(torch.tensor(features).long(), dim=1)

    graphs = [create_sample_graph(key, graph) for _ in range(4)]
    batched_graph = dgl.batch(graphs)  # 使用 DGL 的 batch 函数将多个图合并为一个 batch 图

    # 前向传播，处理 batch 图
    output = model(batched_graph)

    # 打印输出结果
    print("Model Output:", output)
    print("Output Shape:", output.shape)

if __name__ == '__main__':
    # 测试生成的特征
    op_idx = [0, 1, 2, 3, 4]
    features = generate_op_features(op_idx)
    print(features)



