# coding : utf-8
# Author : Yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年06月22日15:57:27）
import platform
import numpy as np
import multiprocessing
from torch.utils.data import DataLoader
from data_provider.data_control import get_dataset, load_data
import pickle

from data_provider.data_scaler import get_scaler

# 数据集定义
class DataModule:
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.x, self.y = load_data(config)
        if config.debug:
            self.x, self.y = self.x[:int(len(self.x) * 0.10)], self.y[:int(len(self.x) * 0.10)]
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, self.x_scaler, self.y_scaler = self.get_split_dataset(self.x, self.y, config)
        self.train_set, self.valid_set, self.test_set = get_dataset(self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, config)
        self.train_loader, self.valid_loader, self.test_loader = self.get_dataloaders(self.train_set, self.valid_set, self.test_set, config)
        config.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')
    
    def preprocess_data(self, x, y, config):
        x = np.array(x).astype(np.float32)
        y = np.array(y).astype(np.float32)
        return x, y
    
    def get_split_dataset(self, x, y, config):
        x, y = self.preprocess_data(x, y, config)
        train_ratio, valid_ratio, _ = parse_split_ratio(config.spliter_ratio)

        if config.use_train_size:
            train_size = int(config.train_size)
        else:
            train_size = int(len(x) * train_ratio)

        if config.eval_set:
            valid_size = int(len(x) * valid_ratio)
        else:
            valid_size = 0

        if config.classification:
            return get_train_valid_test_classification_dataset(x, y, train_size, valid_size, config)
        else:
            
            return get_train_valid_test_dataset_transfer(x, y, train_size, valid_size, config)
        

    def get_dataloaders(self, train_set, valid_set, test_set, config):

        if platform.system() == 'Linux' and 'ubuntu' in platform.version().lower():
            max_workers = multiprocessing.cpu_count() // 4
            prefetch_factor = 4
        else:
            max_workers = 0
            prefetch_factor = None

        train_loader = DataLoader(
            train_set,
            batch_size=config.bs,
            drop_last=False,
            shuffle=True,
            pin_memory=True,
            collate_fn=lambda batch: train_set.custom_collate_fn(batch, config),
            num_workers=max_workers,
            prefetch_factor=prefetch_factor
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=config.bs,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda batch: valid_set.custom_collate_fn(batch, config),
            num_workers=max_workers,
            prefetch_factor=prefetch_factor
        )
        test_loader = DataLoader(
            test_set,
            batch_size=config.bs,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda batch: test_set.custom_collate_fn(batch, config),
            num_workers=max_workers,
            prefetch_factor=prefetch_factor
        )
        return train_loader, valid_loader, test_loader


def parse_split_ratio(ratio_str):
    """解析如 '7:1:2' 的字符串为归一化比例 [0.7, 0.1, 0.2]"""
    parts = list(map(int, ratio_str.strip().split(':')))
    total = sum(parts)
    return [p / total for p in parts]


def get_train_valid_test_dataset(x, y, train_size, valid_size, config):
    if config.shuffle:
        indices = np.random.permutation(len(x))
        x, y = x[indices], y[indices]
    train_x = x[:train_size]
    train_y = y[:train_size]
    valid_x = x[train_size:train_size + valid_size]
    valid_y = y[train_size:train_size + valid_size]
    test_x = x[train_size + valid_size:]
    test_y = y[train_size + valid_size:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_train_valid_test_classification_dataset(x, y, train_size, valid_size, config):
    from collections import defaultdict
    import random
    class_data = defaultdict(list)
    for now_x, now_label in zip(x, y):
        class_data[now_label].append(now_x)
    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x, test_y = [], []
    for label, now_x in class_data.items():
        random.shuffle(now_x)
        train_x.extend(now_x[:train_size])
        train_y.extend([label] * len(now_x[:train_size]))
        valid_x.extend(now_x[train_size:train_size + valid_size])
        valid_y.extend([label] * len(now_x[train_size:train_size + valid_size]))
        test_x.extend(now_x[train_size + valid_size:])
        test_y.extend([label] * len(now_x[train_size + valid_size:]))
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_train_valid_test_dataset_transfer(x, y, train_size, valid_size, config):
    if not config.transfer:
        indices = np.arange(len(x))
        shuffle_indices = np.random.permutation(indices)
        train_idx = shuffle_indices[:train_size]
        valid_idx = shuffle_indices[train_size:train_size + valid_size]
        test_idx = shuffle_indices[train_size + valid_size:]
        with open(f'{config.path}/{config.dataset}_train_idx.pkl','wb') as f:
            pickle.dump(train_idx, f) 
        with open(f'{config.path}/{config.dataset}_valid_idx.pkl','wb') as f:
            pickle.dump(valid_idx, f) 
        with open(f'{config.path}/{config.dataset}_test_idx.pkl','wb') as f:
            pickle.dump(test_idx, f) 
    else:
        with open(f'{config.path}/{config.dataset}_train_idx.pkl','rb') as f:
            train_idx = pickle.load(f)
        with open(f'{config.path}/{config.dataset}_valid_idx.pkl','rb') as f:
            valid_idx = pickle.load(f)
        with open(f'{config.path}/{config.dataset}_test_idx.pkl','rb') as f:
            test_idx = pickle.load(f)

    if config.dataset == 'nnlqp':
        matrix = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    elif config.dataset == 'nasbench201':
        matrix = np.concatenate((x, y), axis=1)
    print(matrix[0, -1])
    # 找到全0行的索引
    null_indices = []
    for i in range(len(matrix)):
        if matrix[i][-1] == 0:
            null_indices.append(i)

    train_temp_idx = []
    for i in range(len(train_idx)):
        if train_idx[i] not in null_indices:
            train_temp_idx.append(train_idx[i])
        else:
            continue
    train_idx = np.array(train_temp_idx)

    valid_temp_idx = []
    for i in range(len(valid_idx)):
        if valid_idx[i] not in null_indices:
            valid_temp_idx.append(valid_idx[i])
        else:
            continue
    valid_idx = np.array(valid_temp_idx)

    test_temp_idx = []
    for i in range(len(test_idx)):
        if test_idx[i] not in null_indices:
            test_temp_idx.append(test_idx[i])
        else:
            continue
    test_idx = np.array(test_temp_idx)

    train_x = x[train_idx]
    train_y = y[train_idx]
    valid_x = x[valid_idx]
    valid_y = y[valid_idx]
    test_x = x[test_idx]
    test_y = y[test_idx]

    print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)
    x_scaler = get_scaler(train_x, config, 'None')
    y_scaler = get_scaler(train_y, config, 'minmax')
    train_x = x_scaler.transform(train_x)
    valid_x = x_scaler.transform(valid_x)
    test_x = x_scaler.transform(test_x)
    
    train_y = y_scaler.transform(train_y).astype(np.float32)
    valid_y = y_scaler.transform(valid_y).astype(np.float32)
    test_y = y_scaler.transform(test_y).astype(np.float32)

    return train_x, train_y, valid_x, valid_y, test_x, test_y, x_scaler, y_scaler