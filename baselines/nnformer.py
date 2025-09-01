import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
from torch import Tensor
import numpy as np 

class SquareReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * F.relu(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        dropout: float = 0.0,
        rel_pos_bias: bool = False,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_size = dim // n_head
        self.scale = math.sqrt(self.head_size)

        self.qkv = nn.Linear(dim, 3 * dim, False)
        self.proj = nn.Linear(dim, dim, False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rel_pos_bias = rel_pos_bias
        # if rel_pos_bias:
        #     self.rel_pos_forward = nn.Embedding(10, self.n_head, padding_idx=9)
        #     self.rel_pos_backward = nn.Embedding(10, self.n_head, padding_idx=9)

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        B, L, C = x.shape

        query, key, value = self.qkv(x).chunk(3, -1)
        query = query.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        key = key.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        score = torch.matmul(query, key.mT) / self.scale
        if self.rel_pos_bias:
            adj = adj.masked_fill(torch.logical_and(adj > 1, adj < 9), 0)
            adj = adj.masked_fill(adj != 0, 1)
            adj = adj.float()
            # pe = torch.stack([adj], dim=1).repeat(1, self.n_head // 1, 1, 1)
            # pe = torch.stack([adj.mT], dim=1).repeat(1, self.n_head // 1, 1, 1)
            # pe = torch.stack([adj, adj.mT], dim=1).repeat(1, self.n_head // 2, 1, 1)
            # pe = torch.stack([adj, adj.mT, adj @ adj, adj.mT @ adj.mT], dim=1)
            pe = torch.stack([adj, adj.mT, adj.mT @ adj, adj @ adj.mT], dim=1)
            pe = pe + torch.eye(L, dtype=adj.dtype, device=adj.device)
            pe = pe.int()

            # pe = (
            #     self.rel_pos_forward(rel_pos) + self.rel_pos_backward(rel_pos.mT)
            # ).permute(0, 3, 1, 2)
            # score = score * (1 + pe)
            score = score.masked_fill(pe == 0, -torch.inf)
        attn = F.softmax(score, dim=-1)
        attn = self.attn_dropout(attn)  # (b, n_head, l_q, l_k)
        x = torch.matmul(attn, value)

        x = x.transpose(1, 2).reshape(B, L, C)
        return self.resid_dropout(self.proj(x))

    def extra_repr(self) -> str:
        return f"n_head={self.n_head}"


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        dropout: float,
        droppath: float,
        rel_pos_bias: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # The larger the dataset, the better rel_pos_bias works
        # probably due to the overfitting of rel_pos_bias
        self.attn = MultiHeadAttention(dim, n_head, dropout, rel_pos_bias=rel_pos_bias)
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()

    def forward(self, x: Tensor, rel_pos: Optional[Tensor] = None) -> Tensor:
        x_ = self.norm(x)
        x_ = self.attn(x_, rel_pos)
        return self.drop_path(x_) + x


class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        out_features: Optional[int] = None,
        act_layer: str = "relu",
        drop: float = 0.0,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, False)
        if act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "leaky_relu":
            self.act = nn.LeakyReLU()
        elif act_layer.lower() == "square_relu":
            self.act = SquareReLU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, False)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GINMlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        out_features: Optional[int] = None,
        act_layer: str = "relu",
        drop: float = 0.0,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, False)
        self.gcn = nn.Linear(in_features, hidden_features, False)
        if act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "leaky_relu":
            self.act = nn.LeakyReLU()
        elif act_layer.lower() == "square_relu":
            self.act = SquareReLU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, False)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        out = self.fc1(x)
        gcn_x1, gcn_x2 = self.gcn(x).chunk(2, dim=-1)
        out = out + torch.cat([adj @ gcn_x1, adj.mT @ gcn_x2], dim=-1)
        out = self.act(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        act_layer: str,
        dropout: float,
        droppath: float,
        gcn: bool = False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        if gcn:
            self.mlp = GINMlp(dim, mlp_ratio, act_layer=act_layer, drop=dropout)
        else:
            self.mlp = Mlp(dim, mlp_ratio, act_layer=act_layer, drop=dropout)
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        x_ = self.norm(x)
        x_ = self.mlp(x_, adj)
        return self.drop_path(x_) + x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        mlp_ratio: float,
        act_layer: str,
        dropout: float,
        droppath: float,
    ):
        super().__init__()
        self.self_attn = SelfAttentionBlock(
            dim, n_head, dropout, droppath, rel_pos_bias=True
        )
        self.feed_forward = FeedForwardBlock(
            dim, mlp_ratio, act_layer, dropout, droppath, gcn=True
        )

    def forward(self, x: Tensor, rel_pos: Tensor, adj: Tensor) -> Tensor:
        x = self.self_attn(x, rel_pos)
        x = self.feed_forward(x, adj)
        return x


# Main class
class Encoder(nn.Module):
    def __init__(
        self,
        depths: List[int] = [12],
        dim: int = 192,
        n_head: int = 4,
        mlp_ratio: float = 4.0,
        act_layer: str = "relu",
        dropout: float = 0.0,
        droppath: float = 0.0,
    ):
        super().__init__()
        self.num_layers = sum(depths)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, droppath, self.num_layers)]

        # Encoder stage
        self.layers = nn.ModuleList()
        for i in range(depths[0]):
            droppath = dpr[i]
            self.layers.append(
                EncoderBlock(dim, n_head, mlp_ratio, act_layer, dropout, droppath)
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, rel_pos: Tensor, adj: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, rel_pos, adj)
        output = self.norm(x)
        return output


class RegHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        avg_tokens: bool = False,
        class_token: bool = True,
        depth_embed: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert (
            avg_tokens or class_token
        ), "`class_token` must be true if `avg_tokens` is false"
        self.avg_tokens = avg_tokens
        self.class_token = class_token
        self.depth_embed = depth_embed
        self.layer = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:  # x(b/n_gpu, l, d)
        if self.avg_tokens:
            if self.class_token:
                x = x[:, 1:]
            if self.depth_embed:
                x = x[:, :-1]
            x_ = x.mean(dim=1)
        else:
            x_ = x[:, 0, :]  # (b, d)

        res = self.layer(x_)
        return res


class NNFormer(nn.Module):
    def __init__(
        self,
        depths: List[int] = [12],
        in_chans: int = 32,
        dim: int = 192,
        n_head: int = 4,
        mlp_ratio: float = 4.0,
        act_layer: str = "relu",
        dropout: float = 0.1,
        droppath: float = 0.0,
        avg_tokens: bool = False,
        class_token: bool = True,
        depth_embed: bool = True,
        dataset: str = "nasbench",
    ):
        super().__init__()

        self.op_embed = nn.Linear(in_chans, dim, False)
        self.depth_embed = nn.Linear(in_chans, dim, False) if depth_embed else None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if class_token else None

        self.norm = nn.LayerNorm(dim)
        self.encoder = Encoder(
            depths=depths,
            dim=dim,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            dropout=dropout,
            droppath=droppath,
        )
        self.head = RegHead(dim, 1, avg_tokens, class_token, depth_embed, dropout)

        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.constant_(m.weight, 0)
            # nn.init.trunc_normal_(m.weight, std=0.02)

    @torch.jit.ignore()
    def no_weight_decay(self):
        no_decay = {}
        return no_decay

    def forward(self, seqcode, rel_pos, depth, adj, static_feats=None) -> Tensor:
        seqcode = self.op_embed(seqcode)
        
        # print(seqcode.shape, rel_pos.shape)
        if self.cls_token is not None:
            seqcode = torch.cat(
                [self.cls_token.expand(seqcode.shape[0], -1, -1), seqcode], dim=1
            )
            new_adj = torch.zeros(
                adj.shape[0], adj.shape[1] + 1, adj.shape[2] + 1, device=adj.device
            )
            new_adj[:, 1:, 1:] = adj
            adj = new_adj
            new_rel_pos = torch.ones(
                rel_pos.shape[0], rel_pos.shape[1] + 1, rel_pos.shape[2] + 1, device=rel_pos.device
            )
            new_rel_pos[:, 1:, 1:] = rel_pos
            rel_pos = new_rel_pos
        # print(seqcode.shape, rel_pos.shape)
            
        if self.depth_embed is not None:
            seqcode = torch.cat([seqcode, self.depth_embed(depth)], dim=1)
            new_adj = torch.zeros(
                adj.shape[0], adj.shape[1] + 1, adj.shape[2] + 1, device=adj.device
            )
            new_adj[:, :-1, :-1] = adj
            adj = new_adj
            new_rel_pos = torch.ones(
                rel_pos.shape[0], rel_pos.shape[1] + 1, rel_pos.shape[2] + 1, device=rel_pos.device
            )
            new_rel_pos[:, :-1, :-1] = rel_pos
            rel_pos = new_rel_pos
        
        # print(seqcode.shape, rel_pos.shape)
        seqcode = self.norm(seqcode)
        aev = self.encoder(seqcode, rel_pos, adj.to(torch.float))
        # multi_stage:aev(b, 1, d)
        predict = self.head(aev) + 0.5
        return predict



class Embedder:
    def __init__(
        self, num_freqs, embed_type="nape", input_type="tensor", input_dims=1
    ):
        self.num_freqs = num_freqs
        self.max_freq = max(32, num_freqs)
        self.embed_type = embed_type
        self.input_type = input_type
        self.input_dims = input_dims
        self.eps = 0.01
        if input_type == "tensor":
            self.embed_fns = [torch.sin, torch.cos]
            self.embed = self.embed_tensor
        else:
            self.embed_fns = [np.sin, np.cos]
            self.embed = self.embed_array
        self.create_embedding_fn()

    def __call__(self, x):
        return self.embed(x)

    def create_embedding_fn(self):
        max_freq = self.max_freq
        N_freqs = self.num_freqs

        if self.embed_type == "nape":
            freq_bands = (
                (self.eps + torch.linspace(1, max_freq, N_freqs)) * math.pi / (max_freq + 1)
            )

        elif self.embed_type == "nerf":
            freq_bands = (
                torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)
            ) * math.pi

        elif self.embed_type == "trans":
            dim = self.num_freqs
            freq_bands = torch.tensor([1 / (10000 ** (j / dim)) for j in range(dim)])

        self.freq_bands = freq_bands
        self.out_dim = self.input_dims * len(self.embed_fns) * len(freq_bands)

    def embed_tensor(self, inputs: Tensor):
        self.freq_bands = self.freq_bands.to(inputs.device)
        return torch.cat([fn(self.freq_bands * inputs) for fn in self.embed_fns], -1)

    def embed_array(self, inputs):
        return np.concatenate([fn(self.freq_bands * inputs) for fn in self.embed_fns])



def tokenizer3(
    ops: List[int], adj, depth: int, dim_x: int = 192, embed_type: str = "nape"
):
    # adj = torch.tensor(adj)
    rel_pos = adj
    if embed_type == "onehot_op":
        code_ops = F.one_hot(torch.tensor(ops), num_classes=dim_x)
        code_depth = F.one_hot(torch.tensor([depth]), num_classes=dim_x)
        return (code_ops.to(torch.int8), adj.to(torch.int8), code_depth.to(torch.int8))
    elif embed_type == "onehot_oppos":
        code_ops = F.one_hot(torch.tensor(ops), num_classes=dim_x // 2)
        code_pos = F.one_hot(torch.arange(len(ops)), num_classes=dim_x // 2)
        code_ops = torch.cat([code_ops, code_pos], dim=-1)
        # # Another implementation
        # code_ops = F.one_hot(torch.tensor(ops), num_classes=dim_x)
        # code_ops[:, dim_x // 2:] = F.one_hot(torch.arrange(len(ops)), num_classes=dim_x // 2)
        code_depth = F.one_hot(torch.tensor([depth]), num_classes=dim_x)
        return (code_ops.to(torch.int8), adj.to(torch.int8), code_depth.to(torch.int8))
    elif embed_type == "onehot_oplaplacian":
        code_ops = F.one_hot(torch.tensor(ops), num_classes=dim_x)
        code_ops[:, dim_x // 2 : dim_x // 2 + adj.shape[-1]] = adj.sum(-1) - adj
        code_depth = F.one_hot(torch.tensor([depth]), num_classes=dim_x)
        return (code_ops.to(torch.int8), adj.to(torch.int8), code_depth.to(torch.int8))
    else:
        # encode operation
        # fn = Embedder(dim_x // 2, embed_type=embed_type)
        # code_ops_list = [fn(torch.Tensor([30]))]
        # code_ops_list += [fn(torch.Tensor([op])) for op in ops]
        # code_ops = torch.stack(code_ops_list, dim=0)  # (len, dim_x)

        # depth = torch.Tensor([depth])
        # code_depth = fn(depth).reshape(1, -1)

        # rel_pos = torch.full((len(ops) + 2, len(ops) + 2), fill_value=9).int()
        # rel_pos[1:-1, 1:-1] = adj
        
        fn = Embedder(dim_x // 2, embed_type=embed_type)
        code_ops = torch.stack([fn(torch.Tensor([op])) for op in ops], dim=0)  # (len(ops), dim_x)
        code_depth = fn(torch.Tensor([depth])).reshape(1, -1)
        # rel_pos = torch.as_tensor(adj, dtype=torch.int32)  # (len(ops), len(ops))
        # print(rel_pos)
        return code_ops, rel_pos, code_depth
    

class NARLoss(nn.Module):
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_rank: float = 0.2,
        lambda_consist: float = 1.0,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_rank = lambda_rank
        self.lambda_consist = lambda_consist
        self.loss_mse = nn.MSELoss()
        self.loss_rank = nn.L1Loss()
        # self.loss_rank = nn.SmoothL1Loss(beta=0.005)
        self.loss_consist = nn.L1Loss()
        # self.loss_consist = nn.SmoothL1Loss(beta=0.005)

    def forward(self, predict: Tensor, target: Tensor):
        loss_mse = self.loss_mse(predict, target) * self.lambda_mse

        index = torch.randperm(predict.shape[0], device=predict.device)
        v1 = predict - predict[index]
        v2 = target - target[index]
        loss_rank = self.loss_rank(v1, v2) * self.lambda_rank
        # v1 = predict.unsqueeze(1) - predict.unsqueeze(0)
        # v2 = target.unsqueeze(1) - target.unsqueeze(0)
        # loss_rank = self.loss_rank(v1, v2) * self.lambda_rank

        loss_consist = 0
        if self.lambda_consist > 0:
            source_pred, aug_pred = predict.chunk(2, 0)
            loss_consist = (
                self.loss_consist(source_pred, aug_pred) * self.lambda_consist
            )
        loss = loss_mse + loss_rank + loss_consist
        return loss

def padding_for_batch3(code, adj):
    MAX_LEN = 7
    if len(adj) < MAX_LEN:
        for i in range(MAX_LEN - len(adj)):
            for l in adj:
                l.append(0)
        adj.extend([[0]*MAX_LEN for _ in range(MAX_LEN - len(adj))])

        code_ = torch.zeros((MAX_LEN, code.shape[1]))
        code_[:code.shape[0], :] = code
        return code_, adj
    else:
        return code, adj