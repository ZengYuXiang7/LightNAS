# coding : utf-8
# Author : Yuxiang Zeng
# 每次开展新实验都改一下这里
from baselines.brpnas import BRPNAS
from baselines.flops import Flops
from baselines.gat import GAT
from baselines.gru import GRU
from baselines.lstm import LSTM
from baselines.narformer import NarFormer
from layers.metric.distance import PairwiseLoss
from exp.exp_base import BasicModel
from modules.TransNAS import TransNAS
from modules.backbone import Backbone


class Model(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_size = config.input_size
        self.hidden_size = config.rank
        if config.model == 'gnn':
            self.model = Backbone(self.input_size, config)
        
        elif config.model == 'flops':
            self.model = Flops(1, config)
        elif config.model == 'flops-mac':
            self.model = Flops(2, config)
            
        elif config.model == 'nn_meter':
            self.model = nn_Meter(self.input_size, config)
        elif config.model == 'nnlqp':
            self.model = NNLQP(self.input_size, config)
            
            
        elif config.model == 'lstm':
            self.model = LSTM(self.input_size, config)  
        elif config.model == 'gru':
            self.model = GRU(self.input_size, config)  
            
            
        elif config.model == 'brp-nas':
            self.model = BRPNAS(self.input_size, config)  
        elif config.model == 'gat':
            self.model = GAT(self.input_size, config)  
            
            
        elif config.model == 'narformer':
            self.model = NarFormer(self.input_size, config)
            
            
        elif config.model == 'ours':
            self.model = TransNAS(self.input_size, config)  
        else:
            raise ValueError(f"Unsupported model type: {config.model}")


        