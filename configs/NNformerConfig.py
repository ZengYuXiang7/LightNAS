# coding: utf-8
# Author: mkw
# Date: 2025-06-08 14:37
# Description: NNformerConfig

from configs.default_config import *
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class NNformerConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = "nnformer"
    verbose: int = 1
    # -------- 数据 / dataset --------
    percent: float = 469
    override_data: bool = False
    finetuning: bool = False

    # -------- DataLoader --------
    batch_size: int = 256
    
    # -------- 网络结构 / encoding --------
    embed_type: str = "nape"                           # nape | nerf | trans
    depths: List[int] = field(default_factory=lambda: [6, 1, 1, 1])
    act_function: str = "relu"
    class_token: bool = True
    depth_embed: bool = True
    enc_dim: int = 96
    in_chans: int = 96
    graph_d_model: int = 160
    graph_n_head: int = 4
    graph_d_ff: int = 640
    d_model: int = 192
    avg_tokens: bool = False

    # -------- dropout --------
    dropout: float = 0.1
    drop_path_rate: float = 0.0

    # -------- device --------
    parallel: bool = False
    device: int = 0

    # -------- optimizer --------
    opt: str = "adamw"
    opt_eps: Optional[float] = None
    opt_betas: Optional[List[float]] = None
    momentum: float = 0.9
    weight_decay: float = 0.01

    # -------- 学习率调度 --------
    sched: str = "cosine"
    lr: float = 1e-3
    lr_cycle_mul: float = 1.0
    min_ratio: float = 1e-1
    decay_rate: float = 0.1
    warmup_lr: float = 1e-6
    warmup_epochs: int = 5
    lr_cycle_limit: int = 1
    epochs: int = 4000
    patience: int = 100

    # -------- 训练控制 --------
    do_train: bool = False
    resume: str = ""
    save_path: str = "model/"
    save_epoch_freq: int = 1000
    pretrained_path: Optional[str] = None

    # -------- EMA --------
    model_ema: bool = True
    model_ema_decay: float = 0.99
    model_ema_force_cpu: bool = True
    model_ema_eval: bool = True

    # -------- 损失函数 --------
    lambda_mse: float = 1.0
    lambda_rank: float = 0.2
    lambda_consistency: float = 0.0
