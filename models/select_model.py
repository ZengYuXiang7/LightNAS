"""按实验配置选择延迟预测模型。"""

import torch

from models.LLGAT import LLGAT
from models.baselines.birnn import BiRNN
from models.baselines.brp_nas import BRP_NAS
from models.baselines.dnnperf import DNNPerf
from models.baselines.flops import Flops
from models.baselines.gnn import GraphSAGEConv
from models.baselines.gru import GRU
from models.baselines.lstm import LSTM
from models.baselines.mlp import MLP


def select_model(args):
    # 2026-11-13 21:53：选择 LLGAT，比较架构图与硬件描述融合的预测效果。
    if args.model == 'ours':
        model = LLGAT(args)
    # 2026-11-13 21:53：选择 BRP-NAS，比较图结构编码的预测效果。
    elif args.model == 'brp_nas':
        model = BRP_NAS(args)
    # 2026-11-13 21:53：选择图卷积基线，比较架构图表示的预测效果。
    elif args.model == 'gcn':
        model = GraphSAGEConv(6, args.rank, 6, args)
    # 2026-11-13 21:53：选择 MLP，比较操作独热编码的预测效果。
    elif args.model == 'mlp':
        model = MLP(8 * 7, args.rank, 1, args)
    # 2026-11-13 21:53：选择 LSTM，比较操作序列编码的预测效果。
    elif args.model == 'lstm':
        model = LSTM(8, args.rank, 1, args)
    # 2026-11-13 21:53：选择 GRU，比较门控序列编码的预测效果。
    elif args.model == 'gru':
        model = GRU(8, args.rank, 1, args)
    # 2026-11-13 21:53：选择双向循环网络，比较双向操作序列编码的预测效果。
    elif args.model == 'birnn':
        model = BiRNN(7, args.rank, 1, args)
    # 2026-11-13 21:53：选择 DNNPerf，比较节点与边信息建模的预测效果。
    elif args.model == 'dnnperf':
        model = DNNPerf(in_node_feats=7, node_hidden_feats=args.rank, in_edge_feats=2, edge_hidden_feats=args.rank, args=args)
    # 2026-11-13 21:53：选择 FLOPs 基线，比较操作编号和计算量的预测效果。
    elif args.model == 'flops':
        model = Flops(args)
    return model


class Model(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.args = args
        self.input_size = data.x.shape[-1]
        self.hidden_size = args.rank
        self.model = select_model(args)

    def forward(self, adjacency, features):
        prediction = self.model.forward(adjacency, features)
        prediction = prediction.flatten()
        return prediction
