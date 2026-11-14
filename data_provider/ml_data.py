"""从统一延迟表准备单设备机器学习实验数据。"""

import numpy as np

from data_provider.data import experiment


def load_ml_data(args):
    # 复用其他模型的设备与架构读取，保留时延的秒单位。
    source = experiment(args)
    measurements = source.load_data(args)
    operations, latency = source.preprocess_data(measurements, args)

    # 输入、六个操作和输出统一为八节点，不添加设备编号。
    features = np.empty((len(operations), 8), dtype=np.int64)
    features[:, 0] = 5
    features[:, 1:7] = operations
    features[:, 7] = 6
    order = np.random.permutation(len(features))
    features = features[order]
    latency = latency[order]

    # 保留训练集和固定大小验证集，测试覆盖全部架构。
    valid_end = args.train_size + args.ml_valid_size
    train_x = features[:args.train_size]
    train_y = latency[:args.train_size]
    valid_x = features[args.train_size:valid_end]
    valid_y = latency[args.train_size:valid_end]
    test_x = features
    test_y = latency
    return train_x, train_y, valid_x, valid_y, test_x, test_y
