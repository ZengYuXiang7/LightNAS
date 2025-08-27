# coding : utf-8
# Author : Yuxiang Zeng
# 每次开展新实验都改一下这里
from baselines.brpnas import BRPNAS
from baselines.flops import Flops
from baselines.gat import GAT
from baselines.gru import GRU
from baselines.lstm import LSTM
from baselines.narformer import NarFormer
from baselines.narformer2 import NarFormer2
from baselines.nnformer import NNFormer
from models.layers.metric.distance import PairwiseLoss
from exp.exp_base import BasicModel
from models.TransNAS import ACLoss, PairwiseDiffLoss, TransNAS
from models.backbone import Backbone


class Model(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_size = config.input_size
        self.hidden_size = config.d_model
        if config.model == 'gnn':
            self.model = Backbone(self.input_size, config)
        
        elif config.model == 'flops':
            self.model = Flops(1, config)
        elif config.model == 'flops-mac':
            self.model = Flops(2, config)
            
        
            
            
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
        
        elif config.model == 'narformer2':
            self.model = NarFormer2(
                dataset=config.dataset,
                feat_shuffle=config.feat_shuffle,
                glt_norm=config.glt_norm,
                n_attned_gnn=config.n_attned_gnn,
                num_node_features=config.num_node_features,
                gnn_hidden=config.gnn_hidden,
                fc_hidden=config.fc_hidden,
                use_degree=config.use_degree,
                norm_sf=config.norm_sf,
                ffn_ratio=config.ffn_ratio,
                real_test=config.real_test
            )
            
        elif config.model == 'nnformer':
            self.model = NNFormer(
                depths=config.depths,
                in_chans=config.in_chans,
                dim=config.graph_d_model,
                n_head=config.graph_n_head,
                mlp_ratio=config.graph_d_ff // config.graph_d_model,
                act_layer=config.act_function,
                dropout=config.dropout,
                droppath=config.drop_path_rate,
                avg_tokens=config.avg_tokens,
                class_token=config.class_token,
                depth_embed=config.depth_embed,
                dataset=config.dataset,
            )
        
            
        elif config.model == 'nnlqp':
            self.model = NNLQP(self.input_size, config)
            
        elif config.model == 'ours':
            # self.model = TransNAS(self.input_size, config)  
            self.model = Backbone(self.input_size, config)
            # self.model = BRPNAS(self.input_size, config)  
            self.rank_loss = PairwiseDiffLoss('l1')
            self.ac_loss = ACLoss('l1')
            
        else:
            raise ValueError(f"Unsupported model type: {config.model}")


        