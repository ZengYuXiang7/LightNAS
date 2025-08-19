# coding : utf-8
# Author : yuxiang Zeng
# 根据需要来改变这里的内容

from data_process.get_latency import *
from data_provider.data_getitem_bench import *


def load_data(config):
    if config.dataset == '201_acc':
        data = get_bench201_acc(config)
    elif config.dataset == 'nnlqp': 
        data = get_nnlqp(config)
    return data


def get_dataset(data, split, config):
    """
    返回一个数据集实例：DatasetClass(data, split, config)
    支持的 model:
      - 'ours'               -> NASDataset
      - 'narformer'          -> SeqDataset
      - 'gat'                -> GraphDataset
      - 'brp-nas'            -> BRPNASDataset
      - 'lstm', 'gru'        -> RNNDataset
      - 'flops', 'flops-mac' -> ProxyDataset
    """
    model = str(getattr(config, "model", "")).lower()
    split = str(split).lower()
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"split must be 'train'/'valid'/'test', got: {split}")

    if model in {"ours"}:
        DatasetClass = OursDataset
    elif model == "2":
        DatasetClass = NASDataset
    elif model == "narformer":
        DatasetClass = SeqDataset
    elif model == "gat":
        DatasetClass = GraphDataset
    elif model == "brp-nas":
        DatasetClass = BRPNASDataset
    elif model in {"lstm", "gru"}:
        DatasetClass = RNNDataset
    elif model in {"flops", "flops-mac"}:
        DatasetClass = ProxyDataset
    else:
        raise NotImplementedError(f"Unsupported model for dataset: {config.model}")

    return DatasetClass(data, split, config)
    