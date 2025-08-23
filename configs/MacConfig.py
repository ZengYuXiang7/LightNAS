from configs.default_config import *
from dataclasses import dataclass



@dataclass
class MacConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'flops-mac'
    # dataset: str = 'nasbench201'
    dataset: str = '201_acc'
    bs: int = 32
    spliter_ratio: str = '1:4:95'
    epochs: int = 200
    patience: int = 50
    num_layers: int = 2
    d_model: int = 32
    ffn_method: str = 'ffn'
    src_dataset: str = 'data/nasbench201/pkl/embedded-gpu-jetson-nono-fp16.pkl'
    dst_dataset: str = 'data/nasbench201/pkl/desktop-gpu-gtx-1080ti-fp32.pkl'
    transfer: bool = False