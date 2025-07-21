# coding : utf-8
# Author : yuxiang Zeng
# 根据需要来改变这里的内容

from data_provider.get_latency import *
from data_provider.data_getitem import *

def load_data(config):
    if config.dataset == 'nasbench201':
        x, y = get_latency_transfer(config)
    elif config.dataset == 'nnmeter': 
        x, y, x_scaler, y_scaler = get_latency(config)
    return x, y


def get_dataset(train_x, train_y, valid_x, valid_y, test_x, test_y, config):
    if config.model in ['narformer', 'transnas']:
        return (
            SeqDataset(train_x, train_y, 'train', config),
            SeqDataset(valid_x, valid_y, 'valid', config),
            SeqDataset(test_x, test_y, 'test', config)
        )
    
    elif config.model == 'gat':
        return (
            GraphDataset(train_x, train_y, 'train', config),
            GraphDataset(valid_x, valid_y, 'valid', config),
            GraphDataset(test_x, test_y, 'test', config)
        )
        
    elif config.model == 'brp-nas':
        return (
            BRPNASDataset(train_x, train_y, 'train', config),
            BRPNASDataset(valid_x, valid_y, 'valid', config),
            BRPNASDataset(test_x, test_y, 'test', config)
        )
        
    elif config.model in ['lstm', 'gru']:
        return (
            RNNDataset(train_x, train_y, 'train', config),
            RNNDataset(valid_x, valid_y, 'valid', config),
            RNNDataset(test_x, test_y, 'test', config)
        )
    elif config.model in ['flops', 'flops-mac']:
        return (
            ProxyDataset(train_x, train_y, 'train', config),
            ProxyDataset(valid_x, valid_y, 'valid', config),
            ProxyDataset(test_x, test_y, 'test', config)
        )
    
    