# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import pickle
import os
from data_provider.data_scaler import get_scaler


def get_nasbench201(config):
    # 只读取一个文件，指定文件config.dst_dataset
    with open(config.dst_dataset, 'rb') as file:
        data = pickle.load(file)
    
    x , y = [], []
    for matrix in data:
        x.append(matrix[:6])
        y.append(matrix[6])
    x = np.array(x).astype(np.int32)
    y = np.array(y).reshape(-1, 1).astype(np.float32)
    return x, y


def get_nnlqp(config):
    if not config.transfer:
        root_dir = './datasets/nnlqp/unseen_structure'
        with open('./datasets/nnlqp/unseen_structure/gt.txt', 'r') as f:
            dataset = f.readlines()
        x, y = [], []
        for line in dataset: #gt.txt
            # model_types.add(line.split()[4])
            line = line.rstrip()
            items = line.split(" ")
            speed_id = str(items[0])
            graph_id = str(items[1])
            batch_size = int(items[2])
            cost_time = float(items[3])
            plt_id = int(items[5])
            x.append(speed_id)
            y.append(cost_time)
        x, y = np.array(x), np.array(y)
    else:
        root_dir = '.datasets/nnlqp/multi_platform/gt.txt'
        with open('./datasets/nnlqp/multi_platform/gt.txt', 'r') as f:
            dataset = f.readlines()
        x, y = [], []
        for line in dataset: #gt.txt
            # model_types.add(line.split()[4])
            line = line.rstrip()
            items = line.split(" ")
            speed_id = str(items[0])
            graph_id = str(items[1])
            batch_size = int(items[2])
            cost_time = float(items[3])
            plt_id = int(items[5])
            x.append(speed_id)
            y.append(cost_time)
        x, y = np.array(x), np.array(y)
    return x, y 
    