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
            self.data = take_subset(self.data, ratio=0.1, seed=config.seed, random_sample=False)
            
        self.train_data, self.valid_data, self.test_data = self.get_split_dataset(self.data, config)
        
        self.x_scaler, self.y_scaler = self.get_scalers(self.train_data, config)
        self.train_data = self.normalize_data(self.train_data, self.x_scaler, self.y_scaler, config)
        self.valid_data = self.normalize_data(self.valid_data, self.x_scaler, self.y_scaler, config)
        self.test_data  = self.normalize_data(self.test_data,  self.x_scaler, self.y_scaler, config)
        
        self.train_set = get_dataset(self.train_data, 'train', config)
        self.valid_set = get_dataset(self.valid_data, 'valid', config)
        self.test_set  = get_dataset(self.test_data,  'test',  config)
        
        self.train_loader = self.build_loader(self.train_set, bs=config.bs, is_train=True)
        self.valid_loader = self.build_loader(self.valid_set, bs=config.bs, is_train=False)
        self.test_loader  = self.build_loader(self.test_set,  bs=config.bs, is_train=False)
        
        config.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')
    

    def get_split_dataset(self, data, config):
        """
        仅切分，不做归一化。
        输入:
            - data: dict[str -> np.ndarray]，各字段第一维长度一致
            - config: 需包含 spliter_ratio（如 '8:1:1'），可选 seed
        返回:
            - train_data, valid_data, test_data  (与 data 同结构的 dict，值是切片后的 np.ndarray)
        """
        assert isinstance(data, dict) and len(data) > 0, "data 必须是非空 dict"

        # 确保全是 np.array，并校验样本数一致
        N = None
        for k, v in data.items():
            if not isinstance(v, np.ndarray):
                v = np.array(v)
                data[k] = v
            if N is None:
                N = len(v)
            else:
                assert len(v) == N, f"字段 {k} 的样本数 {len(v)} 与其他字段不一致（应为 {N})"

        # 解析比例
        ratio_str = config.spliter_ratio
        parts = list(map(int, ratio_str.strip().split(":")))
        total = sum(parts)
        tr, vr, te = [p / total for p in parts]

        # 打乱索引
        seed = config.seed
        rng = np.random.default_rng(seed)
        idx = np.arange(N)
        rng.shuffle(idx)

        # 计算切分点
        n_train = int(N * tr)
        n_valid = int(N * vr)
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train + n_valid]
        test_idx  = idx[n_train + n_valid:]

        # 按索引切片为三个 dict（数组重排后自然是 0..k-1）
        def slice_dict(d, indices):
            out = {}
            for k, v in d.items():
                out[k] = v[indices]
            return out

        train_data = slice_dict(data, train_idx)
        valid_data = slice_dict(data, valid_idx)
        test_data  = slice_dict(data, test_idx)

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
        x_keys = ("flops", "params")
        x_scaler = {}
        for key in x_keys:
            x_scaler[key] = get_scaler(data[key], config, selected_method='stander')
        
        values = np.array(data[config.predict_target], dtype=np.float32)
        y_scaler = get_scaler(values, config, selected_method='stander')
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
        for key, value in data.items():
            if key in x_scaler:
                value = np.array(value, dtype=np.float32)
                normalized_data[key] = x_scaler[key].transform(value)
            elif key == config.predict_target and y_scaler:
                value = np.array(value, dtype=np.float32)
                normalized_data[key] = y_scaler.transform(value)
            else:
                normalized_data[key] = value
        return normalized_data
        
    

    def build_loader(self, dataset, bs, is_train):
        
        print(f"Batch Size {bs}")
        
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
                batch_size=bs,
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
                batch_size=bs,
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
    对所有字段使用同一组索引切片，保证样本对齐：
    - dict[str -> np.ndarray/torch.Tensor/list/tuple]: 统一取前 ratio 部分（或随机抽 ratio 部分）
    - 其他类型：按第0维或序列长度切片
    """

    if ratio is None or ratio >= 1.0:
        return obj

    # 统一做切片的小工具
    def _slice_by_idx(x, idx):
        if isinstance(x, np.ndarray):
            return x[idx]
        if isinstance(x, torch.Tensor):
            return x.index_select(0, torch.as_tensor(idx, device=x.device))
        if isinstance(x, (list, tuple)):
            sel = [x[i] for i in idx]
            return type(x)(sel)
        # 能否用通用索引
        try:
            return x[idx]
        except Exception:
            return x  # 不支持索引就原样返回

    # 计算 n
    def _calc_n(length):
        return max(1, int(length * float(ratio)))

    # dict：对齐切片（核心需求）
    if isinstance(obj, dict):
        # 取样本总数 N（要求各字段第一维一致）
        keys = list(obj.keys())
        assert len(keys) > 0, "空字典无法取子集"
        # 找到第一个可取 len 的字段
        N = None
        for k in keys:
            try:
                N = len(obj[k])
                break
            except Exception:
                continue
        assert N is not None and N > 0, "字典里没有可按第0维切片的字段"

        # 校验第一维一致（尽量只在常见类型上检查）
        for k in keys:
            v = obj[k]
            try:
                lv = len(v)
                assert lv == N, f"字段 {k} 的样本数 {lv} 与其他字段不一致（应为 {N}）"
            except Exception:
                # 无法取 len 的跳过一致性检查
                pass

        n = _calc_n(N)
        if random_sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(N, size=n, replace=False)
            idx.sort()  # 可选：保持原顺序
        else:
            idx = np.arange(n)

        # 对每个字段按同一 idx 切片
        out = {}
        for k in keys:
            out[k] = _slice_by_idx(obj[k], idx)
        return out

    # 非 dict：按自身长度切片
    try:
        N = len(obj)
    except Exception:
        return obj

    n = _calc_n(N)
    if random_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=n, replace=False)
        idx.sort()
    else:
        idx = np.arange(n)
    return _slice_by_idx(obj, idx)

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
