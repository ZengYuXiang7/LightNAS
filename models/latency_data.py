"""NAS-Bench-201 架构字符串、操作编号和固定八节点图之间的转换。"""

import numpy as np


OPERATION_NAMES = ('none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3')


def get_arch_vector_from_arch_str(arch_str):
    # 按官方架构字符串中的边顺序转换六个操作。
    operations = []
    for node in arch_str.split('+'):
        for edge in node.strip('|').split('|'):
            name = edge.split('~')[0]
            operations.append(OPERATION_NAMES.index(name))
    return operations


def get_arch_str_from_arch_vector(arch_vector):
    # 将六个操作编号还原为 NAS-Bench-201 架构字符串。
    names = []
    for operation in arch_vector:
        names.append(OPERATION_NAMES[operation])
    arch_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*names)
    return arch_str


def get_matrix_and_ops(operations):
    # 八节点包含输入、六个操作槽位和输出；none 与 skip 也保留节点。
    adjacency = np.array([
        [0, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int64)
    operation_ids = [5] + list(operations) + [6]
    return adjacency, operation_ids


def get_adjacency_and_features(adjacency, operation_ids):
    # 数据层只输出八节点；自环用于现有图卷积，七类操作保留独立编号。
    graph = adjacency.copy()
    np.fill_diagonal(graph, 1)
    features = np.eye(7, dtype=np.int64)[np.asarray(operation_ids, dtype=np.int64)]
    return graph, features
