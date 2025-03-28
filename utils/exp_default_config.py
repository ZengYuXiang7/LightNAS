# coding : utf-8
# Author : yuxiang Zeng
from dataclasses import dataclass


@dataclass
class LoggerConfig:
    logger: str = 'None'

@dataclass
class ExperimentConfig:
    seed: int = 0
    rounds: int = 1
    epochs: int = 200
    patience: int = 20
    monitor_metrics: str = 'NMAE'

    verbose: int = 10
    device: str = 'cuda'
    debug: bool = False
    record: bool = True
    hyper_search: bool = False
    continue_train: bool = False


@dataclass
class BaseModelConfig:
    model: str = 'ours'
    rank: int = 40
    retrain: bool = True
    num_layers: int = 3


@dataclass
class DatasetInfo:
    path: str = './datasets/'
    dataset: str = 'cpu'
    train_size: int = 100
    density: float = 0.01
    eval_set: bool = True
    shuffle: bool = True
    use_train_size: bool = True
    scaler_method: str = 'minmax'



@dataclass
class TrainingConfig:
    bs: int = 32
    lr: float = 0.001
    decay: float = 0.0001
    loss_func: str = 'L1Loss'
    optim: str = 'AdamW'


@dataclass
class OtherConfig:
    classification: bool = False
    ablation: int = 0
    try_exp: int = -1
