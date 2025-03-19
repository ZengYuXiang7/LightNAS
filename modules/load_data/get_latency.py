# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import pickle

import os

from utils.data_scaler import get_scaler


def get_latency(config):
    file_names = os.listdir(config.path + config.dataset)
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
    y = np.array(y)

    scaler = get_scaler(y, config)
    y = scaler.transform(y).astype(np.float32)
    # x = scaler.transform(x).astype(np.float32)

    return x, y, scaler
