# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import pickle
import os
from data_process.create_latency import get_adjacency_and_features, get_arch_str_from_arch_vector, get_matrix_and_ops
from data_process.nas_201_api import NASBench201API as API
from tqdm import *

def get_bench201_acc(config):
    with open('./data/nasbench201/pkl/desktop-cpu-core-i7-7820x-fp32.pkl', 'rb') as f:
        df = pickle.load(f)
    api = API('./data_process/nas_201_api/NAS-Bench-201-v1_0-e61699.pth', verbose=False)
    import numpy as np 
    data = {
        "key": [],
        "adj_matrix": [],
        "features": [],
        "flops": [],
        "params": [],
        "accuracy": [],
        "latency": [],
    }

    for i in trange(len(df)):
        try:
            key = np.asarray(df[i, :-1], dtype=np.int32)
            arch_str = get_arch_str_from_arch_vector(key)
            index = api.query_index_by_arch(arch_str)

            cost_info = api.get_cost_info(index, dataset='cifar10-valid')
            flops  = float(cost_info['flops'])
            params = float(cost_info['params'])

            adj_matrix, label = get_matrix_and_ops(key)
            adj_matrix, features = get_adjacency_and_features(adj_matrix, label)
            features = np.argmax(features, axis=1)

            acc_info = api.get_more_info(
                index,
                dataset='cifar10-valid',
                iepoch=None,
                hp='200',
                is_random=False
            )

            accuracy = float(acc_info['test-accuracy']) 
            latency = float(df[i, -1])

            # -------- 改成往 list 里 append --------
            data["key"].append(key)
            data["adj_matrix"].append(adj_matrix)
            data["features"].append(features)
            data["flops"].append(flops)
            data["params"].append(params)
            data["accuracy"].append(accuracy)
            data["latency"].append(latency)

        except Exception as e:
            print(f"[Warning] Skipped item {i} due to: {e}")
            
    for key in data:
        data[key] = np.array(data[key])
        print(f"{key} shape: {data[key].shape}")
        
    return data




def get_bench101_acc(config):
    with open('.data/101_acc_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data



def get_nnlqp(config):
    if not config.transfer:
        root_dir = './data/nnlqp/unseen_structure'
        with open('./data/nnlqp/unseen_structure/gt.txt', 'r') as f:
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
        root_dir = '.data/nnlqp/multi_platform/gt.txt'
        with open('./data/nnlqp/multi_platform/gt.txt', 'r') as f:
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
    