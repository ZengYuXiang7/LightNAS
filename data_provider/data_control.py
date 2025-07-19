# coding : utf-8
# Author : yuxiang Zeng
# 根据需要来改变这里的内容

from data_provider.get_latency import *
from data_provider.data_getitem import GraphDataset

def load_data(config):
    if config.dataset == 'nasbench201':
        x, y = get_latency_transfer(config)
    elif config.dataset == 'nnmeter': 
        x, y, x_scaler, y_scaler = get_latency(config)
    return x, y


def get_dataset(train_x, train_y, valid_x, valid_y, test_x, test_y, config):
    return (
        GraphDataset(train_x, train_y, 'train', config),
        GraphDataset(valid_x, valid_y, 'valid', config),
        GraphDataset(test_x, test_y, 'test', config)
    )
    