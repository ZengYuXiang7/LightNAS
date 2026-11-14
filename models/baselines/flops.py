"""使用架构操作与 NAS-Bench-201 官方 FLOPs 预测延迟。"""

import pickle

import torch

from models.latency_data import get_arch_str_from_arch_vector


class Flops(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # 从统一表读取 CIFAR-10 官方 FLOPs，单位为百万次操作。
        with open(args.latency_table, 'rb') as file:
            self.architectures = pickle.load(file)
        self.encoder = torch.nn.Linear(9, args.rank)
        self.linear = torch.nn.Linear(args.rank, 1)

    def forward(self, graph, operations):
        # 按操作编号还原架构键，查询该架构的 FLOPs。
        flops_values = []
        for operation_ids in operations[:, 1:-1].detach().cpu().tolist():
            arch_str = get_arch_str_from_arch_vector(operation_ids)
            flops_values.append(self.architectures[arch_str]['flops'])
        flops = torch.tensor(flops_values, dtype=torch.float32, device=operations.device).unsqueeze(1)
        features = torch.cat((operations.float(), flops), dim=1)
        hidden = self.encoder(features)
        output = self.linear(hidden)
        return output
