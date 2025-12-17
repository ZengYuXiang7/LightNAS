# coding : utf-8
# Author : Yuxiang Zeng
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import torch
import collections
from data_provider.data_center import DataModule
from exp.exp_train import RunOnce
from exp.exp_model import Model
import exp.exp_efficiency
import utils.utils

torch.set_default_dtype(torch.float32)


if __name__ == '__main__':
    # Experiment Settings, logger, plotter
    from utils.exp_logger import Logger
    from utils.exp_metrics_plotter import MetricsPlotter
    from utils.utils import set_settings
    from utils.exp_config import get_config

    # config = get_config('FlopsConfig')
    # config = get_config('MacConfig')
    # config = get_config('LSTMConfig')
    # config = get_config('GRUConfig')
    # config = get_config('BRPNASConfig')
    # config = get_config('GATConfig')
    # config = get_config('NarFormerConfig')
    # config = get_config('NarFormer2Config')
    # config = get_config('NNformerConfig')
    config = get_config("OurModelConfig")
    set_settings(config)
    
    