# coding: utf-8
# Author: mkw
# Date: 2025-06-08 14:37
# Description: SeasonalTrendModelConfig

from configs.default_config import *
from dataclasses import dataclass, field



@dataclass
class OurModelConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    # 模型基本参数
    model: str = 'ours'
    bs: int = 256
    spliter_ratio: str = '5:4:91'
    input_size: int = 192
    
    epochs: int = 1500
    patience: int = 100
    d_model: int = 60 
    lp_d_model: int = 16
    optim: str = 'AdamW'
    
    num_layers: int = 2
    num_heads: int = 4
    
    monitor_reverse: bool = True
    monitor_metric: str = 'KendallTau'
    
    predict_target: str = 'accuracy'  #  latency accuracy 

    # Transformer 结构
    # att_method: str = 'rms'
    # norm_method: str = 'rms'
    # ffn_method: str = 'ffn'
    
    gcn_method: str = 'gcn'
    norm_method: str = 'batch'
    ffn_method: str = 'ffn'

    # 数据集路径
    transfer: bool = False
    
    # 位置编码相关
    embed_type: str = 'nerf'
    multires_x: int = 32
    multires_r: int = 32
    multires_p: int = 32
    
    rank_loss: bool = True  # 
    ac_loss: bool = True  # 
    
