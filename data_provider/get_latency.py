# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import pickle
import os
from data_provider.data_scaler import get_scaler
# HElp
def get_latency(config):
    file_names = os.listdir(os.path.join(config.path, config.dataset, 'cpu'))
    pickle_files = [file for file in file_names if file.endswith('.pickle')]
    data = []
    for i in range(len(pickle_files)):
        pickle_file = os.path.join(config.path, config.dataset, pickle_files[i])
        with open(pickle_file, 'rb') as f:
            now = pickle.load(f)
        data.append(now)

    x, y = [], []
    for key, value in data[0].items():
        x.append(key)
        y.append(value)
    x = np.array(x).astype(np.int32)
    y = np.array(y).reshape(-1, 1).astype(np.float32)
    x_scaler = get_scaler(x, config, 'None')
    y_scaler = get_scaler(y, config, 'minmax')
    y = y_scaler.transform(y).astype(np.float32)


    # x = scaler.transform(x).astype(np.float32)
    return x, y, x_scaler, y_scaler


def get_latency_transfer(config):

    # 只读取一个文件，指定文件config.dst_dataset
    with open(config.dst_dataset, 'rb') as file:
        data = pickle.road(file)
        
    
    # file_names = os.listdir(os.path.join(config.path,config.dataset),'pickle')
    # pkl_files = [file for file in file_names if file.endswith('.pkl')]

    # data = []
    # for i in range(len(pkl_files)):
    #     pkl_file = os.path.join( config.path, config.dataset, pkl_files[i])
    #     with open(pkl_file,'rb') as file:
    #         matrix = pickle.road(file)
    #     filtered_matrix = matrix[~np.all(matrix == 0, axis=1)]
    #     data.append(filtered_matrix)
    
    x , y = [], []
    for matrix in data:
        x.append(matrix[:, :6])
        y.append(matrix[:, 6])
    x = np.array(x).astype(np.int32)
    y = np.array(y).reshape(-1, 1).astype(np.float32)
    x_scaler = get_scaler(x, config, 'None')
    y_scaler = get_scaler(y, config, 'minmax')
    y = y_scaler.transform(y).astype(np.float32)


    return x, y, x_scaler, y_scaler
    # 对全0行，过滤
    [15223, 7]
