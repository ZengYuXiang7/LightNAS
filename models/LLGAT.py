# coding : utf-8
# Author : Anonymous

import torch
import dgl
from dgl.nn.pytorch import SAGEConv
import pickle
from pathlib import Path

from models.graph_utils import add_global_dgl


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, rank, args):
        super(Encoder, self).__init__()
        self.args = args
        # 五种架构操作、输入、输出和全局节点各有独立编号。
        if args.op_encoder == 'embed':
            self.dnn_embedding = torch.nn.Embedding(8, rank)
        elif args.op_encoder == 'value':
            self.transfer = torch.nn.Linear(1, rank)
        elif args.op_encoder == 'one_hot':
            self.transfer = torch.nn.Linear(8, rank)


    def forward(self, features):
        if self.args.op_encoder == 'embed':
            feats = self.dnn_embedding(features).reshape(features.shape[0] * 9, -1)
        elif self.args.op_encoder == 'one_hot':
            bs = features.shape[0]
            feats = torch.nn.functional.one_hot(features, num_classes=8).float().reshape(bs * 9, -1)
            feats = self.transfer(feats)
        elif self.args.op_encoder == 'value':
            bs = features.shape[0]
            feats = features / 7
            feats = feats.reshape(bs * 9, 1)
            feats = self.transfer(feats)
        else:
            raise NotImplementedError
        return feats


class GraphEncoder(torch.nn.Module):
    def __init__(self, rank, order, args):
        super(GraphEncoder, self).__init__()
        self.args = args
        self.rank = rank
        self.order = order
        if args.graph_encoder == 'gcn':
            self.layers = torch.nn.ModuleList([dgl.nn.pytorch.GraphConv(rank, rank) for i in range(order)])
        elif args.graph_encoder == 'graphsage':
            self.layers = torch.nn.ModuleList([dgl.nn.pytorch.SAGEConv(rank, rank, aggregator_type='gcn') for i in range(order)])
        elif args.graph_encoder == 'gat':
            self.layers = torch.nn.ModuleList([dgl.nn.pytorch.GATConv(rank, rank, args.heads, 0.10) for i in range(order)])
        else:
            raise NotImplementedError
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(rank) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(order)])
        self.dropout = torch.nn.Dropout(0.10)

    def forward(self, graph, features):
        g, feats = graph, features
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats)
            if self.args.graph_encoder == 'gat':
                feats = feats.mean(dim=1)  # 聚合多个头的输出
            feats = norm(feats)
            feats = act(feats)
            if self.args.graph_encoder != 'gat':
                feats = self.dropout(feats)
        batch_sizes = torch.as_tensor(g.batch_num_nodes()).to(self.args.device)  # 每个图的节点数
        first_nodes_idx = torch.cumsum(torch.cat((torch.tensor([0]).to(self.args.device), batch_sizes[:-1])), dim=0)  # 使用torch.cat来连接Tensor
        first_node_features = feats[first_nodes_idx]
        return first_node_features


class LLMs(torch.nn.Module):
    def __init__(self, args):
        super(LLMs, self).__init__()
        self.args = args
        # 从独立设备表读取硬件向量，设备名不存在时直接报错。
        hardware_path = Path(__file__).resolve().parent.parent / 'data' / 'agu' / 'devices.pickle'
        with hardware_path.open('rb') as file:
            devices = pickle.load(file)
        self.aug_data = devices[args.device_name]
        # CPU 增加核心类型、L2 缓存、内存通道和标称功耗；GPU 使用 MHz 加速频率。
        if args.dataset == 'cpu':
            norm = [7, 32, 32, 40, 100, 16, 32, 40, 4, 250]
        else:
            norm = [9000, 2000, 16, 400]
        for i in range(len(norm)):
            self.aug_data[i] /= norm[i]
        self.aug_data = torch.tensor(self.aug_data).float().unsqueeze(0)
        # Encoder 
        self.info_encoder = torch.nn.Linear(len(norm), 100)

    def forward(self, batch_size):
        aug_data_repeated = self.aug_data.expand(batch_size, -1).to(self.args.device)
        info_embeds = self.info_encoder(aug_data_repeated)
        return info_embeds


class Predictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Predictor, self).__init__()
        self.NeuCF = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim // 2),  # FFN
            torch.nn.LayerNorm(hidden_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 2),  # FFN
            torch.nn.LayerNorm(hidden_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(hidden_dim // 2, output_dim)  # y
        )

    def forward(self, x):
        y = self.NeuCF(x)
        return y.flatten()


class LLGAT(torch.nn.Module):
    def __init__(self, args):
        super(LLGAT, self).__init__()
        self.args = args
        self.rank = args.rank
        self.op_encoder = Encoder(9, args.rank, self.args)
        self.graph_encoder = GraphEncoder(args.rank, args.order, self.args)
        self.llms = LLMs(args)

        if args.Ablation in [0, 3]:
            predictor_dim = args.rank * 2
        elif args.Ablation in [2, 4, 6]:
            predictor_dim = args.rank * 10
        else:
            predictor_dim = args.rank

        self.predictor = Predictor(predictor_dim, self.rank, 1)

    def forward(self, graph, key):
        # 保留八个架构节点，在前向中加入全局节点供图汇聚。
        graph, key = add_global_dgl(graph, key, bidirectional=True)
        if not self.args.Ablation:
            op_embeds = self.op_encoder(key)
            graph_embeds = self.graph_encoder(graph, op_embeds)
            info_embeds = self.llms(key.shape[0])
            embeds = torch.cat([graph_embeds, info_embeds], dim=1)
            y = self.predictor(embeds)
        elif self.args.Ablation == 1:
            op_embeds = self.op_encoder(key)
            graph_embeds = self.graph_encoder(graph, op_embeds)
            y = self.predictor(graph_embeds)
        elif self.args.Ablation == 2:
            op_embeds = self.op_encoder(key).reshape(key.shape[0], -1)
            info_embeds = self.llms(key.shape[0])
            embeds = torch.cat([op_embeds, info_embeds], dim=1)
            y = self.predictor(embeds)
        elif self.args.Ablation == 3:
            # 但是这里是One-hot
            op_embeds = self.op_encoder(key)
            graph_embeds = self.graph_encoder(graph, op_embeds)
            info_embeds = self.llms(key.shape[0])
            embeds = torch.cat([graph_embeds, info_embeds], dim=1)
            y = self.predictor(embeds)
        elif self.args.Ablation == 4:
            op_embeds = self.op_encoder(key).reshape(key.shape[0], -1)
            y = self.predictor(op_embeds)
        elif self.args.Ablation == 5:
            # 但是这里是One-hot
            op_embeds = self.op_encoder(key)
            graph_embeds = self.graph_encoder(graph, op_embeds)
            y = self.predictor(graph_embeds)
        elif self.args.Ablation == 6:
            # 但是这里是One-hot
            op_embeds = self.op_encoder(key).reshape(key.shape[0], -1)
            info_embeds = self.llms(key.shape[0])
            embeds = torch.cat([op_embeds, info_embeds], dim=1)
            y = self.predictor(embeds)
        return y

