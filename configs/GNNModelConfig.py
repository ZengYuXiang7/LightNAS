# coding: utf-8
# Author: mkw
# Date: 2025-06-08 14:37
# Description: SeasonalTrendModelConfig

from configs.default_config import *
from dataclasses import dataclass

from configs.MainConfig import OtherConfig


@dataclass
class GNNModelConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'gnn'
    dataset: str = 'nasbench201'
    bs: int = 32
    rank: int = 32
    spliter_ratio: str = '1:3:26'
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    num_layers: int = 2
    revin: bool = True
    d_model: int = 64
    kernel_size: int = 25
    individual: bool = True
    gcn_method: str = 'gat'
    norm_method: str = 'batch'
    ffn_method: str = 'ffn'
