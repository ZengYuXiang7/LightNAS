# coding: utf-8
# Author: mkw
# Date: 2025-06-08 14:37
# Description: SeasonalTrendModelConfig

from configs.default_config import *
from dataclasses import dataclass, field

from configs.MainConfig import OtherConfig


@dataclass
class TransModelConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    # 模型基本参数
    model: str = 'ours'
    dataset: str = 'nasbench201'
    bs: int = 16
    spliter_ratio: str = '5:4:91'
    epochs: int = 600
    patience: int = 50
    num_layers: int = 2
    num_heads: int = 2
    
    input_size: int = 192

    d_model: int = 192

    # Transformer 结构
    att_method: str = 'rms'
    norm_method: str = 'rms'
    ffn_method: str = 'ffn'

    # 数据集路径
    src_dataset: str = 'datasets/nasbench201/pkl/embedded-gpu-jetson-nono-fp16.pkl'
    dst_dataset: str = 'datasets/nasbench201/pkl/desktop-gpu-gtx-1080ti-fp32.pkl'
    transfer: bool = False

    # 位置编码相关
    embed_type: str = 'nerf'
    multires_x: int = 32
    multires_r: int = 32
    multires_p: int = 32
