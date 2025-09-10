
# coding : utf-8
# Author : Yuxiang Zeng
import random
import torch
from baselines.narformer import *
from models.layers.transformer import Transformer
# self.encoder = Transformer(config.d_model, config.num_heads, config.num_layers, 'rms', 'ffn', 'self')
import torch
import torch.nn as nn
import einops




# coding : utf-8
# Author : Yuxiang Zeng
import random
import torch
from baselines.narformer import *
from models.layers.encoder.discrete_enc import DiscreteEncoder
from models.layers.transformer import Transformer
# self.encoder = Transformer(config.d_model, config.num_heads, config.num_layers, 'rms', 'ffn', 'self')
import torch
import torch.nn as nn




class TransNAS(nn.Module):
    def __init__(self, enc_in, config):
        super(TransNAS, self).__init__()
        self.config = config
        self.d_model = config.d_model
        # self.op_embedding = torch.nn.Embedding(7, config.d_model)
        self.op_embedding = DiscreteEncoder(num_operations=7, encoding_dim=self.d_model, encoding_type=config.op_encoder, output_dim=self.d_model)
        self.tokenGT = TokenGT(c_node=self.d_model, d_model=self.d_model, nhead=config.num_heads, num_layers=config.num_layers, lap_dim=config.lp_d_model, use_edge=False)
        self.fc = nn.Linear(config.d_model, 1)  # 回归或分类

    def forward(self, graphs, features, P):
        features = self.op_embedding(features)
        cls_out, _ = self.tokenGT(graphs, features, P)  # [B, d_model]
        y = self.fc(cls_out)                           # 回归或分类
        return y
    

class TokenGT(nn.Module):
    def __init__(self, c_node: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4, lap_dim: int = 8, use_edge: bool = False):
        super().__init__()
        self.use_edge = use_edge
        self.lap_dim = lap_dim

        # 节点 token 投影
        self.node_proj = nn.Linear(c_node + 2 * lap_dim, d_model)

        # 边 token 投影 (用常数 + LapPE(u)+LapPE(v))
        if use_edge:
            self.edge_proj = nn.Linear(1 + 2 * lap_dim, d_model)

        # type embedding (0=node, 1=edge)
        self.type_embed = nn.Embedding(2, d_model)

        # [graph] special token
        self.graph_tok = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer 编码器
        # enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        # self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        self.encoder = Transformer(d_model, num_layers, nhead, 'rms', 'ffn', 'self')

    def forward(self, adj: torch.Tensor, node_feats: torch.Tensor, P: torch.Tensor):
        B, n, _ = adj.shape
        
        # P = laplacian_node_ids_from_adj(adj, self.lap_dim)  # [B,n,lap_dim]

        # node tokens: [Xv, Pv, Pv]
        node_tok = self.node_proj(torch.cat([node_feats, P, P], dim=-1))  # [B,n,d]
        node_tok = node_tok + self.type_embed(torch.zeros(n, dtype=torch.long, device=adj.device)).unsqueeze(0)
        tokens = [node_tok]

        if self.use_edge:
            edge_tok_per_b = [None] * B
            edge_len = torch.zeros(B, dtype=torch.long, device=adj.device)
            for b in range(B):
                Ab = (adj[b] > 0)
                Ab = Ab.clone()
                Ab.fill_diagonal_(False)
                src, dst = torch.triu(Ab, diagonal=1).nonzero(as_tuple=True)
                m = src.numel()
                edge_len[b] = m
                if m == 0:
                    continue
                Pu, Pv = P[b, src], P[b, dst]
                edge_input = torch.cat([torch.ones(m, 1, device=adj.device), Pu, Pv], dim=-1)  # [m,1+2*lap]
                etok = self.edge_proj(edge_input).unsqueeze(0)                                  # [1,m,d]
                etok = etok + self.type_embed(torch.ones(m, dtype=torch.long, device=adj.device)).unsqueeze(0)
                edge_tok_per_b[b] = etok

            max_m = int(edge_len.max().item() if edge_len.numel() else 0)
            
            if max_m > 0:
                padded = torch.zeros(B, max_m, self.node_proj.out_features, device=adj.device)
                pad_mask_edges = torch.ones(B, max_m, dtype=torch.bool, device=adj.device)
                for b, etok in enumerate(edge_tok_per_b):
                    if etok is None:
                        continue
                    L = etok.size(1)
                    padded[b, :L] = etok[0]
                    pad_mask_edges[b, :L] = False
                tokens.append(padded)
            else:
                pad_mask_edges = torch.zeros(B, 0, dtype=torch.bool, device=adj.device)
                
        else:
            pad_mask_edges = torch.zeros(B, 0, dtype=torch.bool, device=adj.device)

        graph_tok = self.graph_tok.expand(B, -1, -1)  # [B,1,d]
        seq = torch.cat([graph_tok] + tokens, dim=1)  # [B, 1+n(+max_m), d]

        pad_mask = torch.cat([
            torch.zeros(B, 1 + n, dtype=torch.bool, device=adj.device),
            pad_mask_edges
        ], dim=1)

        x_enc = self.encoder(seq, key_padding_mask=pad_mask)
        graph_repr = x_enc[:, 0, :]
        return graph_repr, x_enc
    


class PairwiseDiffLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        """
        :param loss_type: 'l1', 'l2', or 'kldiv'
        """
        super(PairwiseDiffLoss, self).__init__()
        loss_type = loss_type.lower()
        if loss_type == 'l1':
            self.base_loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'kldiv':
            self.base_loss = nn.KLDivLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, predicts, targets):
        """
        :param predicts: Tensor of shape (B,) or (B, 1)
        :param targets: Tensor of shape (B,) or (B, 1)
        :return: Pairwise difference loss
        """
        # 自动 squeeze 支持 [B, 1] 输入
        if predicts.ndim == 2 and predicts.shape[1] == 1:
            predicts = predicts.squeeze(1)
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        if predicts.ndim != 1 or targets.ndim != 1:
            raise ValueError("Both predicts and targets must be 1D tensors.")

        B = predicts.size(0)
        idx = list(range(B))
        random.shuffle(idx)
        shuffled_preds = predicts[idx]
        shuffled_targs = targets[idx]

        diff_preds = predicts - shuffled_preds
        diff_targs = targets - shuffled_targs

        return self.base_loss(diff_preds, diff_targs)
    
    
    
class ACLoss(nn.Module):
    def __init__(self, loss_type='l1', reduction='mean'):
        """
        Architecture Consistency Loss
        :param loss_type: 'l1' or 'l2'
        :param reduction: 'mean' or 'sum'
        """
        super(ACLoss, self).__init__()
        if loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, predictions):
        """
        :param predictions: Tensor of shape [2B, 1] or [2B]
                            where front B are source, back B are augmented
        :return: scalar loss
        """
        N = predictions.shape[0]
        B = N // 2  # 自动 floor 向下取整

        # 仅使用前 B 和后 B，丢弃中间多出的一个（若存在）
        source = predictions[:B]
        augmented = predictions[-B:]
        
        
        # Ensure shape is [B]
        source = source.squeeze(-1) if source.ndim == 2 and source.shape[1] == 1 else source
        augmented = augmented.squeeze(-1) if augmented.ndim == 2 and augmented.shape[1] == 1 else augmented

        return self.criterion(source, augmented)
    
    
    
    
class VIT(nn.Module):
    def __init__(self, enc_in, config):
        super(VIT, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.seq_len = 9  # 如果是 9 个 token（例如来自 NAS cell）

        # 输入 embedding 映射
        self.enc_embedding = nn.Linear(enc_in, self.d_model)

        # 可学习的 cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # 可学习的位置编码：cls_token + patch_token
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len + 1, self.d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.num_heads,
            dim_feedforward=self.d_model * 2,
            batch_first=True,
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # 输出层（cls token → 预测值）
        self.fc = nn.Linear(self.d_model, 1)

        self.dropout = nn.Dropout(0.10)

        # 初始化
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x.shape: [B, seq_len, enc_in]
        B, _, _ = x.shape
        
        x = self.enc_embedding(x)  # [B, seq_len, d_model]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, seq_len+1, d_model]
        x = x + self.pos_embedding                     # 加位置编码

        x = self.dropout(x)
        x = self.encoder(x)                            # 经过 transformer encoder

        cls_out = x[:, 0]                              # 取 cls_token 的输出
        y = self.fc(cls_out)                           # 回归或分类

        return y
    
    
    # def transfer(self, x)
# 
        # return y
        


class PairwiseDiffLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        """
        :param loss_type: 'l1', 'l2', or 'kldiv'
        """
        super(PairwiseDiffLoss, self).__init__()
        loss_type = loss_type.lower()
        if loss_type == 'l1':
            self.base_loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'kldiv':
            self.base_loss = nn.KLDivLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, predicts, targets):
        """
        :param predicts: Tensor of shape (B,) or (B, 1)
        :param targets: Tensor of shape (B,) or (B, 1)
        :return: Pairwise difference loss
        """
        # 自动 squeeze 支持 [B, 1] 输入
        if predicts.ndim == 2 and predicts.shape[1] == 1:
            predicts = predicts.squeeze(1)
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        if predicts.ndim != 1 or targets.ndim != 1:
            raise ValueError("Both predicts and targets must be 1D tensors.")

        B = predicts.size(0)
        idx = list(range(B))
        random.shuffle(idx)
        shuffled_preds = predicts[idx]
        shuffled_targs = targets[idx]

        diff_preds = predicts - shuffled_preds
        diff_targs = targets - shuffled_targs

        return self.base_loss(diff_preds, diff_targs)
    
    
    
class ACLoss(nn.Module):
    def __init__(self, loss_type='l1', reduction='mean'):
        """
        Architecture Consistency Loss
        :param loss_type: 'l1' or 'l2'
        :param reduction: 'mean' or 'sum'
        """
        super(ACLoss, self).__init__()
        if loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, predictions):
        """
        :param predictions: Tensor of shape [2B, 1] or [2B]
                            where front B are source, back B are augmented
        :return: scalar loss
        """
        N = predictions.shape[0]
        B = N // 2  # 自动 floor 向下取整

        # 仅使用前 B 和后 B，丢弃中间多出的一个（若存在）
        source = predictions[:B]
        augmented = predictions[-B:]
        
        
        # Ensure shape is [B]
        source = source.squeeze(-1) if source.ndim == 2 and source.shape[1] == 1 else source
        augmented = augmented.squeeze(-1) if augmented.ndim == 2 and augmented.shape[1] == 1 else augmented

        return self.criterion(source, augmented)