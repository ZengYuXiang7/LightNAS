from configs.default_config import *
from dataclasses import dataclass



@dataclass
class GRUConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'gru'
    bs: int = 32
    spliter_ratio: str = '1:4:95'
    epochs: int = 200
    patience: int = 50
    d_model: int = 64  
    
    src_dataset: str = 'data/nasbench201/pkl/embedded-gpu-jetson-nono-fp16.pkl'
    dst_dataset: str = 'data/nasbench201/pkl/desktop-gpu-gtx-1080ti-fp32.pkl'
    transfer: bool = False
