"""模型前向计算中添加全局节点。"""

import dgl
import torch


def add_global_dgl(graph, operation_ids, bidirectional):
    # 每个八节点图各自添加首位全局节点，再重新组成批次。
    graphs = []
    for component in dgl.unbatch(graph):
        source, target = component.edges()
        nodes = torch.arange(1, component.num_nodes() + 1, device=source.device, dtype=source.dtype)
        global_source = torch.zeros(component.num_nodes() + 1, device=source.device, dtype=source.dtype)
        global_target = torch.arange(component.num_nodes() + 1, device=source.device, dtype=source.dtype)
        source = torch.cat((source + 1, global_source))
        target = torch.cat((target + 1, global_target))
        if bidirectional:
            source = torch.cat((source, nodes))
            target = torch.cat((target, torch.zeros_like(nodes)))
        expanded_graph = dgl.graph((source, target), num_nodes=component.num_nodes() + 1, device=graph.device)
        graphs.append(expanded_graph)
    expanded_batch = dgl.batch(graphs)
    global_ids = torch.full((operation_ids.shape[0], 1), 7, dtype=operation_ids.dtype, device=operation_ids.device)
    expanded_operations = torch.cat((global_ids, operation_ids), dim=1)
    return expanded_batch, expanded_operations


def add_global_dense(adjacency, features):
    # 八节点矩阵与七类操作扩展为九节点矩阵及八类操作。
    batch_size, node_count, _ = adjacency.shape
    expanded_adjacency = adjacency.new_zeros((batch_size, node_count + 1, node_count + 1))
    expanded_adjacency[:, 1:, 1:] = adjacency
    expanded_adjacency[:, 0, :] = 1
    expanded_features = features.new_zeros((batch_size, node_count + 1, features.shape[-1] + 1))
    expanded_features[:, 0, -1] = 1
    expanded_features[:, 1:, :-1] = features
    return expanded_adjacency, expanded_features
