# coding : utf-8
# Author : yuxiang Zeng
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    bs: int = 16
    lr: float = 0.001
    decay: float = 0.0001
    loss_func: str = 'L1Loss'  # L1Loss  MSELoss
    optim: str = 'AdamW'
    epochs: int = 200
    patience: int = 30
    verbose: int = 10
    device: str = 'cuda'
    monitor_metric: str = 'NMAE'
    use_amp: bool = False


@dataclass
class BaseModelConfig:
    model: str = 'ours'
    rank: int = 40
    retrain: bool = True
    num_layers: int = 3


@dataclass
class DatasetInfo:
    path: str = './datasets'
    dataset: str = 'weather'
    train_size: int = 500
    use_train_size: bool = False
    density: float = 0.70
    eval_set: bool = True
    shuffle: bool = False
    scaler_method: str = 'minmax'
    spliter_ratio: str = '7:1:2'


@dataclass
class ExperimentConfig:
    seed: int = 0
    rounds: int = 5
    debug: bool = False
    record: bool = True
    hyper_search: bool = False
    continue_train: bool = False


@dataclass
class LoggerConfig:
    logger: str = 'zyx'