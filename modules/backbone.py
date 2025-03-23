# coding : utf-8
# Author : Yuxiang Zeng
import torch

from layers.encoder.graph_enc import GnnFamily
from layers.encoder.position_enc import PositionEncoding
from layers.encoder.seq_enc import SeqEncoder
from layers.encoder.token_emc import TokenEmbedding
from layers.transformer import Transformer
from modules.temporal_enc import TemporalEmbedding


class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.rank = config.rank

        self.op_embedding = torch.nn.Embedding(7, self.rank)
        self.device_embedding = torch.nn.Parameter(torch.randn(1, self.rank))
        self.gcn = GnnFamily(d_model=self.rank, order=config.num_layers, gcn_method=config.gcn_method, norm_method=config.norm_method, ffn_method=config.ffn_method)
        self.fc = torch.nn.Linear(config.rank, 1)

    def forward(self, graph, op_idx):
        op_embeds = self.op_embedding(op_idx).reshape(op_idx.shape[0] * 9, -1) + self.device_embedding
        graph_embeds = self.gcn(graph, op_embeds)
        y = self.fc(graph_embeds).sigmoid().squeeze(-1)
        return y
