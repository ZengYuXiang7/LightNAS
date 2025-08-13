from configs.default_config import *
from dataclasses import dataclass




@dataclass
class NNMeterConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'nn_meter'
    dataset: str = 'nasbench201'
    bs: int = 32  
    spliter_ratio: str = '1:4:95'  
    epochs: int = 200  
    patience: int = 50  
    verbose: int = 1  
    kernel_size: int = 25
    individual: bool = True
    gcn_method: str = 'gat'
    norm_method: str = 'batch'
    ffn_method: str = 'ffn'
    src_dataset: str = 'datasets/nasbench201/pkl/embedded-gpu-jetson-nono-fp16.pkl'
    dst_dataset: str = 'datasets/nasbench201/pkl/desktop-gpu-gtx-1080ti-fp32.pkl'
    transfer: bool = False

    
    # 设备配置部分
    device_addr: str = None  # 设备地址
    separate_process: bool = False  # 是否在分离的进程中运行
    spawn_server: bool = False  # 是否启动服务器
    device_user: str = None  # 设备用户
    target_args: dict = field(default_factory=lambda: {
        'runs': 100,  # 运行次数
        'avg_between': 10,  # 测量间隔
        'interval': 0.01,  # 时间或频率上的间隔
        'device': 'cpu'  # 默认使用CPU
    })
