# coding : utf-8
# Author : Yuxiang Zeng
import random
import torch
from models.layers.encoder.discrete_enc import DiscreteEncoder
from models.layers.transformer import Transformer
import torch
import torch.nn as nn
import torch.nn.init as init


class TransNAS(nn.Module):
    def __init__(self, enc_in, config):
        super(TransNAS, self).__init__()
        self.config = config
        self.d_model = config.d_model
        
        # self.op_embedding = torch.nn.Embedding(7, config.d_model)
        self.op_embedding = DiscreteEncoder(
            num_operations=7,
            encoding_dim=self.d_model,
            encoding_type=config.op_encoder,
            output_dim=self.d_model,
        )
        self.indeg_embedding = DiscreteEncoder(
            num_operations=10,
            encoding_dim=self.d_model,
            encoding_type="embedding",
            output_dim=self.d_model,
        )
        self.outdeg_embedding = DiscreteEncoder(
            num_operations=10,
            encoding_dim=self.d_model,
            encoding_type="embedding",
            output_dim=self.d_model,
        )

        self.att_bias = SPDSpatialBias(num_heads=config.num_heads, max_dist=99)

        self.tokenGT = TokenGT(
            c_node=self.d_model,
            d_model=self.d_model,
            lap_dim=config.lp_d_model,
            use_edge=False,
        )

        self.encoder = Transformer(
            self.d_model,
            config.num_layers,
            config.num_heads,
            "rms",
            "ffn",
            config.att_method,
        )
        # enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        # self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.pred_head = nn.Linear(config.d_model, 1)  # 回归或分类

    def forward(self, graphs, features, eigvec, indgree, outdegree, dij):
        # seq_embeds = self.op_embedding(features)
        seq_embeds = (
            self.op_embedding(features)
            + self.indeg_embedding(indgree)
            + self.outdeg_embedding(outdegree)
        )

        # [B, seq_len, d_model]
        seq_embeds, att_mask = self.tokenGT(graphs, seq_embeds, eigvec)

        # 获得距离嵌入
        if self.config.att_bias:
            att_bias = self.att_bias(dij)  # [B, H, N, N]
        else:
            att_bias = None

        cls_out = self.encoder(seq_embeds, att_bias)[:, 0, :]  # [B, d_model]

        y = self.pred_head(cls_out)  # 回归或分类

        return y


class SPDSpatialBias(nn.Module):
    """
    把最短路径矩阵 [B, N, N] 编成逐头加性偏置 [B, H, N, N]
    """

    def __init__(self, num_heads: int, max_dist: int):
        super().__init__()
        # 每个距离值 d -> 每个 head 的一个标量
        self.spatial_pos_encoder = nn.Embedding(100, num_heads)

    def forward(self, spatial_pos: torch.Tensor) -> torch.Tensor:
        """
        spatial_pos: [B, N, N], 取值范围 [0, max_dist]
        return: attn_bias [B, H, N, N]
        """
        # [B, N, N, H] -> [B, H, N, N]
        spatial_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)

        B, H, N, N = spatial_bias.shape
        with_cls_spatial_bias = torch.zeros(
            B, H, N + 1, N + 1, device=spatial_bias.device, dtype=spatial_bias.dtype
        )
        with_cls_spatial_bias[:, :, 1:, 1:] = spatial_bias

        return with_cls_spatial_bias


class TokenGT(nn.Module):
    def __init__(
        self,
        c_node: int,
        d_model: int = 128,
        lap_dim: int = 8,
        use_edge: bool = False,
        lap_node_id_sign_flip=True,
    ):
        super().__init__()
        self.use_edge = use_edge
        self.lap_dim = lap_dim
        self.lap_node_id_sign_flip = lap_node_id_sign_flip

        # 节点 token 投影
        # self.node_proj = nn.Linear(c_node + 2 * lap_dim, d_model)
        self.lap_encoder = nn.Linear(1 * lap_dim, d_model, bias=True)

        # 边 token 投影 (用常数 + LapPE(u)+LapPE(v))
        if use_edge:
            self.edge_proj = nn.Linear(1 + 2 * lap_dim, d_model)

        # type embedding (0=node, 1=edge)
        self.type_embed = nn.Embedding(2, d_model)

        # [graph] special token
        self.graph_tok = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(
        self,
        adj: torch.Tensor,
        node_feats: torch.Tensor,
        eigvec: torch.Tensor,
        num_nodes=None,
    ):
        """
        adj:        [B, n, n]
        node_feats: [B, n, xdim]
        P:          [B, n, lap_dim]  (外部已算好 LapPE；如需内部计算，可自行打开注释)
        """
        B, n, _ = adj.shape
        device = adj.device
        node_tok = node_feats + self.lap_encoder(eigvec)  # [B, n, d]

        tokens = [node_tok]  # 先放节点

        pad_mask_edges = torch.zeros(B, 0, dtype=torch.bool, device=device)

        # 图级 token
        graph_tok = self.graph_tok.expand(B, -1, -1)  # [B, 1, d]
        seq_embeds = torch.cat([graph_tok] + tokens, dim=1)  # [B, 1+n(+max_m), d]

        # key padding mask：前 1+n 个位置（图级 + 节点）都为有效(False)；边用 pad_mask_edges
        pad_mask = torch.cat(
            [torch.zeros(B, 1 + n, dtype=torch.bool, device=device), pad_mask_edges],
            dim=1,
        )  # [B, 1+n(+max_m)]

        return seq_embeds, pad_mask


class PairwiseDiffLoss(nn.Module):
    def __init__(self, loss_type="l1"):
        """
        :param loss_type: 'l1', 'l2', or 'kldiv'
        """
        super(PairwiseDiffLoss, self).__init__()
        loss_type = loss_type.lower()
        if loss_type == "l1":
            self.base_loss = nn.L1Loss()
        elif loss_type == "l2":
            self.base_loss = nn.MSELoss()
        elif loss_type == "kldiv":
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
    def __init__(self, loss_type="l1", reduction="mean"):
        """
        Architecture Consistency Loss
        :param loss_type: 'l1' or 'l2'
        :param reduction: 'mean' or 'sum'
        """
        super(ACLoss, self).__init__()
        if loss_type == "l1":
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == "l2":
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
        source = (
            source.squeeze(-1) if source.ndim == 2 and source.shape[1] == 1 else source
        )
        augmented = (
            augmented.squeeze(-1)
            if augmented.ndim == 2 and augmented.shape[1] == 1
            else augmented
        )

        return self.criterion(source, augmented)


class SoftRankLoss(torch.nn.Module):
    def __init__(self, config):
        super(SoftRankLoss, self).__init__()
        self.config = config

    # Kendall
    def diffkendall_tau(self, x, y, alpha=10.0, jitter=0.0):
        x = x.view(-1)
        y = y.view(-1)
        if jitter > 0:
            x = x + jitter * torch.randn_like(x)
            y = y + jitter * torch.randn_like(y)
        n = x.numel()

        dx = x.unsqueeze(0) - x.unsqueeze(1)  # [n,n]
        dy = y.unsqueeze(0) - y.unsqueeze(1)

        sx = torch.tanh(alpha * dx)
        sy = torch.tanh(alpha * dy)

        mask = torch.triu(
            torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1
        )
        concord = (sx * sy)[mask].sum()

        num_pairs = torch.tensor(n * (n - 1) / 2, device=x.device, dtype=concord.dtype)
        return concord / num_pairs

    def diffkendall_loss(self, pred, target, alpha=10.0, jitter=0.0):
        return 1.0 - self.diffkendall_tau(pred, target, alpha, jitter=jitter)

    # Spearman
    def soft_rank(self, v, tau=1.6):
        v = v.view(-1, 1)
        P = torch.sigmoid((v.T - v) / tau)  # [n,n]
        r = 1.0 + P.sum(dim=1)  # 近似秩
        return (r - r.mean()) / (r.std() + 1e-8)

    def spearman_loss(self, a, b, tau=1.0):
        ra, rb = self.soft_rank(a, tau), self.soft_rank(b, tau)
        rho = (ra * rb).mean()
        return 1.0 - rho

    def forward(self, preds, labels):
        """
        :param preds: Tensor of shape (B,) or (B, 1)
        :param labels: Tensor of shape (B,) or (B, 1)
        :return: Rank loss
        """
        # 自动 squeeze 支持 [B, 1] 输入
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        if preds.ndim != 1 or labels.ndim != 1:
            raise ValueError("Both preds and labels must be 1D tensors.")

        loss_spearman = self.spearman_loss(preds, labels, tau=1.6)
        loss_kendall = self.diffkendall_loss(preds, labels, alpha=10, jitter=0)

        return loss_spearman, loss_kendall
