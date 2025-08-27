# coding: utf-8
# Author: mkw
# Date: 2025-06-08 14:37
# Description: SeasonalTrendModelConfig

from configs.default_config import *
from dataclasses import dataclass, field




@dataclass
class NarFormer2Config(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    # 模型基本参数
    model: str = 'narformer2'
    bs: int = 16
    spliter_ratio: str = '5:4:91'
    # spliter_ratio: str = '80:10:10'
    epochs: int = 600
    patience: int = 50
    num_layers: int = 2
    num_heads: int = 2
    
    feat_shuffle: bool = False           # 是否打乱特征
    glt_norm: str = "LN"                 # 图层归一化方式
    n_attned_gnn: int = 6                # GNN 注意力层数
    num_node_features: int = 128         # 节点输入特征维度
    gnn_hidden: int = 128                # GNN 隐层维度
    fc_hidden: int = 128                 # 全连接层隐层维度
    use_degree: bool = True              # 是否使用节点度特征
    norm_sf: str = None                  # 结构特征归一化
    ffn_ratio: int = 4                   # FFN 扩展倍数
    real_test: bool = False              # 是否只进行测试
