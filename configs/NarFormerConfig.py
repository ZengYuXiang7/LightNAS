# coding: utf-8
# Author: mkw
# Date: 2025-06-08 14:37
# Description: SeasonalTrendModelConfig

from configs.default_config import *
from dataclasses import dataclass, field




@dataclass
class NarFormerConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    # 模型基本参数
    model: str = 'narformer'
    dataset: str = 'nnlqp' # nasbench201 nnlqp
    bs: int = 16
    # spliter_ratio: str = '5:4:91'
    spliter_ratio: str = '80:10:10'
    epochs: int = 600
    patience: int = 50
    num_layers: int = 2
    num_heads: int = 2
    
    input_size: int = 192
    graph_d_model: int = 192
    d_model: int = 192
    # 图结构相关
    graph_n_head: int = 6
    graph_d_ff: int = 768
    
    
    # Transformer 结构
    depths: list = field(default_factory=lambda: [6, 1, 1, 1])  # ← 修复此处的 mutable default

    # 数据集路径
    src_dataset: str = 'data/nasbench201/pkl/desktop-cpu-core-i7-7820x-fp32.pkl'
    dst_dataset: str = 'data/nasbench201/pkl/desktop-cpu-core-i7-7820x-fp32.pkl'
    transfer: bool = False

    # 位置编码相关
    embed_type: str = 'nerf'
    multires_x: int = 32
    multires_r: int = 32
    multires_p: int = 32

    # 激活函数与 token 设置
    act_function: str = 'relu'
    use_extra_token: bool = False
    avg_tokens: bool = False
    drop_path_rate: float = 0
    dropout: float = 0

    