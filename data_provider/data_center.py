# coding : utf-8
# Author : Yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年06月22日15:57:27）
import platform
import multiprocessing
from torch.utils.data import DataLoader
from data_provider.data_control import get_dataset, load_data
import pickle

from data_provider.data_scaler import get_scaler
import numpy as np
from torch.utils.data import Sampler


# 数据集定义
class DataModule:
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.data = load_data(config)
        
        if config.debug:
            data = take_subset(data, ratio=0.1, seed=config.seed, random_sample=False)
            
        self.train_data, self.valid_data, self.test_data = self.get_split_dataset(self.data, config)
        
        
        self.x_scaler, self.y_scaler = self.get_scalers(self.train_data, config)
        self.train_data = self.normalize_data(self.train_data, self.x_scaler, self.y_scaler, config)
        self.valid_data = self.normalize_data(self.valid_data, self.x_scaler, self.y_scaler, config)
        self.test_data  = self.normalize_data(self.test_data,  self.x_scaler, self.y_scaler, config)
        
        self.train_set = get_dataset(self.train_data, 'train', config)
        self.valid_set = get_dataset(self.valid_data, 'valid', config)
        self.test_set  = get_dataset(self.test_data,  'test',  config)
        
        self.train_loader = self.build_loader(self.train_set, is_train=True)
        self.valid_loader = self.build_loader(self.valid_set, is_train=False)
        self.test_loader  = self.build_loader(self.test_set,  is_train=False)
        
        config.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')
    

    def get_split_dataset(self, data, config):
        """
        仅切分数据，不做归一化。
        输入:
            - data: dict[int -> sample_dict]
            - config: 需要包含 split_ratio（如 '7:1:2'），seed（可选）
        返回:
            - train_data, valid_data, test_data
        """
        assert isinstance(data, dict) and len(data) > 0, "data 必须是非空 dict"

        # 解析如 '7:1:2' 的字符串为比例 [0.7, 0.1, 0.2]
        ratio_str = config.spliter_ratio
        parts = list(map(int, ratio_str.strip().split(':')))
        total = sum(parts)
        tr, vr, te = [p / total for p in parts]

        # 随机打乱并切分
        seed = getattr(config, "seed", 0)
        rng = np.random.default_rng(seed)
        keys = list(data.keys())
        n = len(keys)
        idx = np.arange(n)
        rng.shuffle(idx)

        n_train = int(n * tr)
        n_valid = int(n * vr)
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train + n_valid]
        test_idx  = idx[n_train + n_valid:]

        # 重建字典，key 从 0 递增
        def rebuild(d, key_list, indices):
            sel = [key_list[i] for i in indices]
            return {i: d[k] for i, k in enumerate(sel)}

        train_data = rebuild(data, keys, train_idx)
        valid_data = rebuild(data, keys, valid_idx)
        test_data  = rebuild(data, keys, test_idx)

        return train_data, valid_data, test_data
    
    def get_scalers(self, data: dict, config):
        """
        为指定的字段训练归一化器，并以 dict 形式返回。
        
        输入:
            data: dict[int -> sample_dict]，例如 {0: {"flops":..., "params":..., "accuracy":...}, ...}
            keys: 需要归一化的字段名
        返回:
            scalers: dict[str -> StandardScaler]
        """
        x_scaler = {}
        x_keys = ("flops", "params")
        for key in x_keys:
            values = [sample[key] for sample in data.values()]
            values = np.array(values, dtype=np.float32)
            scaler = get_scaler(values, config, selected_method='minmax')
            x_scaler[key] = scaler
        
        values = [sample[config.predict_target] for sample in data.values()]
        values = np.array(values, dtype=np.float32)
        y_scaler = get_scaler(values, config, selected_method='none')
        return x_scaler, y_scaler
    
    def normalize_data(self, data, x_scaler, y_scaler, config):
        """
        对数据进行归一化处理。
        
        输入:
            data: dict[int -> sample_dict]，例如 {0: {"flops":..., "params":..., "accuracy":...}, ...}
            x_scaler: dict[str -> StandardScaler]，用于归一化输入特征
            y_scaler: StandardScaler，用于归一化目标值
        返回:
            normalized_data: dict[int -> sample_dict]，归一化后的数据
        """
        normalized_data = {}
        for idx, sample in data.items():
            normalized_sample = {}
            for key, value in sample.items():
                if key in x_scaler:
                    value = np.array(value, dtype=np.float32).reshape(1, -1)
                    normalized_sample[key] = x_scaler[key].transform(value).reshape(-1)
                elif key == config.predict_target and y_scaler:
                    value = np.array(value, dtype=np.float32).reshape(1, -1)
                    normalized_sample[key] = y_scaler.transform(value).reshape(-1)
                else:
                    normalized_sample[key] = value
            normalized_data[idx] = normalized_sample
        return normalized_data
        
    

    def build_loader(self, dataset, is_train):

        if platform.system() == 'Linux' and 'ubuntu' in platform.version().lower():
            max_workers = min(multiprocessing.cpu_count() // 3, 12)
            prefetch_factor = 4  # 可调至 6，建议不要超过 8
        else:
            max_workers = 0
            prefetch_factor = None

        if self.config.dataset == 'nnlqp' and self.config.model != 'flops':
            sampler = FixedLengthBatchSampler(
                data_source=dataset,
                dataset='nnlqp',
                batch_size=self.config.bs,
                include_partial=True,
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                pin_memory=True,
                num_workers=max_workers,
                prefetch_factor=prefetch_factor
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.bs,
                shuffle=is_train,
                drop_last=False,
                pin_memory=True,
                collate_fn=lambda batch: dataset.custom_collate_fn(batch, self.config),
                num_workers=max_workers,
                prefetch_factor=prefetch_factor
            )


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



def take_subset(obj, ratio=0.1, seed=0, random_sample=False):
    """
    通用取子集：
    - dict: 取前 n 个 key（或随机挑 n 个 key），返回新 dict
    - list/tuple: 取前 n 个（或随机挑 n 个），保持类型
    - np.ndarray / torch.Tensor: 沿第 0 维取前 n（或随机挑 n 个）
    其他类型：直接原样返回
    """
    import numpy as np
    import torch

    if ratio is None or ratio >= 1.0:
        return obj

    # 计算 n
    def _n(length):
        return max(1, int(length * ratio))

    # dict
    if isinstance(obj, dict):
        keys = list(obj.keys())
        n = _n(len(keys))
        if random_sample:
            rng = np.random.default_rng(seed)
            sel = rng.choice(keys, size=n, replace=False).tolist()
        else:
            sel = keys[:n]
        return {k: obj[k] for k in sel}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        n = _n(len(obj))
        if random_sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(obj), size=n, replace=False).tolist()
            subset = [obj[i] for i in idx]
        else:
            subset = obj[:n]
        return type(obj)(subset)

    # numpy array
    if isinstance(obj, np.ndarray):
        n = _n(len(obj))
        if random_sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(obj), size=n, replace=False)
            return obj[idx]
        else:
            return obj[:n]

    # torch tensor
    if isinstance(obj, torch.Tensor):
        n = _n(len(obj))
        if random_sample:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(obj), generator=g)[:n]
            return obj.index_select(0, idx)
        else:
            return obj[:n]

    # 其他类型：不处理
    return obj

#https://blog.csdn.net/jokerxsy/article/details/109733852
import numpy as np
import torch
from torch.utils.data import Sampler
class FixedLengthBatchSampler(Sampler):

    def __init__(self, data_source, dataset, batch_size, include_partial=False, rng=None, maxlen=None,
                 length_to_size=None):
        self.data_source = data_source
        self.dataset = dataset
        self.active = False
        if rng is None:
            rng = np.random.RandomState(seed=11)
        self.rng = rng
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.include_partial = include_partial
        self.length_to_size = length_to_size
        self._batch_size_cache = { 0: self.batch_size }
        self.length_map = self.get_length_map()
        self.reset()
        
    def get_length_map(self):
        '''
        Create a map of {length: List[example_id]} and maintain how much of
        each list has been seen.
        '''
        # Record the lengths of each example.
        length_map = {} #{70:[0, 23, 3332, ...], 110:[3, 421, 555, ...], length:[dataidx_0, dataidx_1, ...]}
        for i in range(len(self.data_source)):
            # if self.dataset == 'nnlqp':
            # length = self.data_source[i][0]['netcode'].shape[0]
            length = self.data_source[i][0].shape[0]
            # elif self.dataset == 'nasbench201' or 'nasbench101':
                # length = self.data_source[i][0][0].shape[0] if len(self.data_source[i])==2 else self.data_source[i][0].shape[0]
            if self.maxlen is not None and self.maxlen > 0 and length > self.maxlen:
                continue
            length_map.setdefault(length, []).append(i)
        return length_map

    def get_batch_size(self, length):
        if self.length_to_size is None:
            return self.batch_size
        if length in self._batch_size_cache:
            return self._batch_size_cache[length]
        start = max(self._batch_size_cache.keys())
        batch_size = self._batch_size_cache[start]
        for n in range(start+1, length+1):
            if n in self.length_to_size:
                batch_size = self.length_to_size[n]
            self._batch_size_cache[n] = batch_size
        return batch_size

    def reset(self):
        """

        If include_partial is False, then do not provide batches that are below
        the batch_size.

        If length_to_size is set, then batch size is determined by length.

        """
        # Shuffle the order.
        for length in self.length_map.keys():
            self.rng.shuffle(self.length_map[length])

        # Initialize state.
        state = {} #e.g. {70(length):{'nbatches':3(num_batch), 'surplus':True, 'position':-1}}
        for length, arr in self.length_map.items():
            batch_size = self.get_batch_size(length)
            nbatches = len(arr) // batch_size
            surplus = len(arr) % batch_size
            state[length] = dict(nbatches=nbatches, surplus=surplus, position=-1)

        # Batch order, in terms of length.
        order = [] #[70, 70, 70, 110, ...] length list
        for length, v in state.items():
            order += [length] * v['nbatches']

        ## Optionally, add partial batches.
        if self.include_partial:
            for length, v in state.items():
                if v['surplus'] >= torch.cuda.device_count():
                    order += [length]

        self.rng.shuffle(order)

        self.length_map = self.length_map
        self.state = state
        self.order = order
        self.index = -1

    def get_next_batch(self):
        index = self.index + 1
        length = self.order[index]
        batch_size = self.get_batch_size(length)
        position = self.state[length]['position'] + 1
        start = position * batch_size
        batch_index = self.length_map[length][start:start+batch_size]

        self.state[length]['position'] = position
        self.index = index
        return batch_index

    def __iter__(self):
        self.reset()
        for _ in range(len(self)):
            yield self.get_next_batch()

    def __len__(self):
        return len(self.order)
