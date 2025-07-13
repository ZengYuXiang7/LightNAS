# coding : utf-8
# Author : yuxiang Zeng

import copy
import torch
import numpy as np


def get_arch_vector_from_arch_str(arch_str):
    """
        Args:
            arch_str : a string representation of a cell architecture,
                for example '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_cotorch.nnect~0|nor_conv_3x3~1|skip_cotorch.nnect~2|'
    """
    _opname_to_index = {
        'none': 0,
        'skip_cotorch.nnect': 1,
        'nor_conv_1x1': 2,
        'nor_conv_3x3': 3,
        'avg_pool_3x3': 4,
        'input': 5,
        'output': 6,
        'global': 7
    }

    _opindex_to_name = {value: key for key, value in _opname_to_index.items()}
    nodes = arch_str.split('+')
    nodes = [node[1:-1].split('|') for node in nodes]
    nodes = [[op_and_input.split('~')[0] for op_and_input in node] for node in nodes]

    # arch_vector is equivalent to a decision vector produced by autocaml when using Nasbench201 backend
    arch_vector = [_opname_to_index[op] for node in nodes for op in node]
    return arch_vector


def get_arch_str_from_arch_vector(arch_vector):
    _opname_to_index = {
        'none': 0,
        'skip_cotorch.nnect': 1,
        'nor_conv_1x1': 2,
        'nor_conv_3x3': 3,
        'avg_pool_3x3': 4,
        'input': 5,
        'output': 6,
        'global': 7
    }
    _opindex_to_name = {value: key for key, value in _opname_to_index.items()}
    ops = [_opindex_to_name[opindex] for opindex in arch_vector]
    return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*ops)

def get_matrix_and_ops(g, prune=True, keep_dims=True):
    ''' Return the adjacency matrix and label vector.

        Args:
            g : should be a point from Nasbench102 search space
            prune : remove dangling nodes that only connected to zero ops
            keep_dims : keep the original matrix size after pruning
    '''

    matrix = np.zeros((8, 8), dtype=int)
    labels = [None for _ in range(8)]
    labels[0] = 'input'
    labels[-1] = 'output'
    matrix[0][1] = matrix[0][2] = matrix[0][4] = 1
    matrix[1][3] = matrix[1][5] = 1
    matrix[2][6] = 1
    matrix[3][6] = 1
    matrix[4][7] = 1
    matrix[5][7] = 1
    matrix[6][7] = 1

    for idx, op in enumerate(g):
        if op == 0:  # zero
            matrix[:, idx + 1] = 0
            matrix[idx + 1, :] = 0
        elif op == 1:  # skip-connection
            to_del = []
            for other in range(8):
                if matrix[other, idx + 1]:
                    for other2 in range(8):
                        if matrix[idx + 1, other2]:
                            matrix[other, other2] = 1
                            matrix[other, idx + 1] = 0
                            to_del.append(other2)
            for d in to_del:
                matrix[idx + 1, d] = 0
        else:
            labels[idx + 1] = str(op)

    if prune:
        visited_fw = np.zeros(8, dtype=bool)
        visited_bw = np.zeros(8, dtype=bool)

        def bfs(beg, vis, con_f):
            q = [beg]
            vis[beg] = True
            index = 0  # Initialize index for accessing elements in the queue
            while index < len(q):
                v = q[index]  # Access element without popping
                index += 1  # Increment index instead of popping
                for other in range(8):
                    if not vis[other] and con_f(v, other):
                        q.append(other)
                        vis[other] = True

        bfs(0, visited_fw, lambda src, dst: matrix[src, dst])  # forward
        bfs(7, visited_bw, lambda src, dst: matrix[dst, src])  # backward

        for v in range(7, -1, -1):
            if not visited_fw[v] or not visited_bw[v]:
                labels[v] = None
                if keep_dims:
                    matrix[v, :] = 0
                    matrix[:, v] = 0
                else:
                    matrix = np.delete(matrix, v, axis=0)
                    matrix = np.delete(matrix, v, axis=1)

        if not keep_dims:
            labels = list(filter(lambda l: l is not None, labels))

        assert visited_fw[-1] == visited_bw[0]
        assert visited_fw[-1] == False or matrix.size > 0

        verts = matrix.shape[0]
        assert verts == len(labels)
        for row in matrix:
            assert len(row) == verts

    return matrix, labels


def get_adjacency_and_features(matrix, labels):
    global_row = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1])
    new_graph = np.zeros((9, 9))
    new_graph[0] = global_row
    new_graph[1:, 1:] = matrix
    # 2024 年 8 月 19 日 去自环
    np.fill_diagonal(new_graph, 1)
    matrix = new_graph
    # Create features matrix from labels
    features = np.zeros((matrix.shape[0], 6), dtype=int)
    features[0][5] = 1  # global
    features[1][3] = 1  # input
    features[-1][4] = 1  # output

    for idx, op in enumerate(labels):
        if op is not None and op != 'input' and op != 'output':
            features[idx + 1][int(float(op)) - 2] = 1

    return matrix, features