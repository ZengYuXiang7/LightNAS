# coding : utf-8
# Author : yuxiang Zeng

from utils.exp_default_config import *
from dataclasses import dataclass


@dataclass
class TestConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 128
    device: str = 'cuda'
    epochs: int = 500
    patience: int = 50
    verbose: int = 10
    try_exp: int = 1
    num_layers: int = 32

    norm_method: str = 'batch'
    ffn_method: str = 'ffn'
    gcn_method: str = 'gat'

    multi_dataset: bool = False
    idx: int = 0


@dataclass
class MLPConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mlp'
    bs: int = 32
    rank: int = 32
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    num_layers: int = 2



@dataclass
class RNNConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'rnn'
    bs: int = 128
    rank: int = 32
    epochs: int = 200
    patience: int = 50
    verbose: int = 1

@dataclass
class LSTMConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'lstm'
    bs: int = 128
    rank: int = 32
    epochs: int = 200
    patience: int = 50
    verbose: int = 1

@dataclass
class GRUConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'gru'
    bs: int = 128
    rank: int = 50
    epochs: int = 200
    patience: int = 50
    verbose: int = 1