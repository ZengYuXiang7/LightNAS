# coding: utf-8
# Author: mkw
# Date: 2025-06-08 14:37
# Description: SeasonalTrendModelConfig

from configs.default_config import *
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class OurModelConfig(
    ExperimentConfig,
    BaseModelConfig,
    LoggerConfig,
    DatasetInfo,
    TrainingConfig,
    OtherConfig,
):
    # 模型基本参数
    model: str = "ours"
    # loss_func: str = "L1Loss"  # L1Loss  MSELoss

    epochs: int = 2000
    verbose: int = epochs
    bs: int = 256
    spliter_ratio: str = "5:4:91"

    patience: int = 200

    # model
    op_encoder: str = "embedding"  # embedding, onehot, discrete
    input_size: int = 192
    d_model: int = 256
    lp_d_model: int = 8
    optim: str = "Adam"

    #
    num_layers: int = 6
    num_heads: int = 4

    monitor_reverse: bool = True
    monitor_metric: str = "KendallTau"
    predict_target: str = "accuracy"  #  latency accuracy

    # Transformer 结构
    att_method: str = "self"
    att_bias: bool = True  # 是否使用距离嵌入

    # 数据集路径
    transfer: bool = False

    rank_loss: bool = True  #
    ac_loss: bool = True  #

    try_exp: int = 1  # 1-8
