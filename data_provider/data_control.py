# coding : utf-8
# Author : yuxiang Zeng
# 根据需要来改变这里的内容

from data_process.get_latency import *
from data_provider.data_getitem_bench import *
import pickle


def load_data(config):
    try:
        with open(f"./data/{config.dataset}_data.pkl", "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        if config.dataset == "201_acc":
            data = get_bench201_acc(config)
        elif config.dataset == "101_acc":
            data = get_bench101_acc(config)
        elif config.dataset == "nnlqp":
            data = get_nnlqp(config)
        with open(f"./data/{config.dataset}_data.pkl", "wb") as f:
            pickle.dump(data, f)
    return data


def get_dataset(data, split, config):
    """
    返回一个数据集实例：DatasetClass(data, split, config)
    支持的 model:
      - 'ours'               -> NASDataset
    """
    dataset = config.dataset
    split = str(split).lower()
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"split must be 'train'/'valid'/'test', got: {split}")

    DatasetClass = NasBenchDataset

    return DatasetClass(data, split, config)
