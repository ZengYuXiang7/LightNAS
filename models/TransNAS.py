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
            num_operations=8 if self.config.dataset != "nnlqp" else 34,
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

        self.lap_encoder = nn.Linear(1 * config.lp_d_model, self.d_model, bias=True)
        self.att_bias = SPDSpatialBias(num_heads=config.num_heads, max_dist=99)

        # [graph] special token
        self.graph_tok = nn.Parameter(torch.randn(1, 1, self.d_model))

        self.encoder = Transformer(
            self.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            norm_method="rms",
            ffn_method="ffn",
            att_method=config.att_method,
        )
        # enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        # self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 回归或分类
        self.pred_head = nn.Sequential(
            nn.Linear(config.d_model, 1),
            # nn.Dropout(0.10)
        )

    def forward(
        self, graphs, features, eigvec, indgree, outdegree, dij, key_padding_mask
    ):
        B, _ = features.shape

        # 首先给节点加入拓扑信息
        seq_embeds = (
            self.op_embedding(features)
            + self.indeg_embedding(indgree)
            + self.outdeg_embedding(outdegree)
        )

        # 给节点加入相对位置信息？
        seq_embeds = seq_embeds + self.lap_encoder(eigvec)  # [B, n, d]

        # [B, seq_len, d_model]
        # 图级 token
        graph_tok = self.graph_tok.expand(B, -1, -1)  # [B, 1, d]
        seq_embeds = torch.cat([graph_tok] + [seq_embeds], dim=1)  # [B, 1+n(+max_m), d]

        # 获得距离嵌入
        if self.config.att_bias:
            attn_mask = self.att_bias(dij)  # [B, H, N, N]
        else:
            attn_mask = None

        # 因为加了CLS所以多了一位
        # B, L, _ = seq_embeds.shape   # 这里 L = max_len
        # device = seq_embeds.device
        # cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)  # [B, 1], False
        # key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)  # [B, 8]
        # print(seq_embeds.shape, attn_mask.shape, key_padding_mask.shape)

        cls_out = self.encoder(seq_embeds, attn_mask)[:, 0, :]  # [B, d_model]

        y = self.pred_head(cls_out)  # 回归或分类

        return y


class SPDSpatialBias(nn.Module):
    """
    把最短路径矩阵 [B, N, N] 编成逐头加性偏置 [B, H, N, N]
    """

    def __init__(self, num_heads: int, max_dist: int):
        super().__init__()
        # 每个距离值 d -> 每个 head 的一个标量
        self.spatial_pos_encoder = nn.Embedding(245, num_heads)

    def forward(self, spatial_pos: torch.Tensor) -> torch.Tensor:
        """
        spatial_pos: [B, N, N]
        return: attn_bias [B, H, N+1, N+1]
        """
        # [B, N, N, H]
        spatial_bias = self.spatial_pos_encoder(spatial_pos)

        # ---- mask 掉取值为 14 的位置 ----
        # mask = (spatial_pos == 14).unsqueeze(-1)  # [B, N, N, 1]
        # spatial_bias = spatial_bias.masked_fill(mask, float("-inf"))

        # -> [B, H, N, N]
        spatial_bias = spatial_bias.permute(0, 3, 1, 2)

        B, H, N, _ = spatial_bias.shape

        # ---- 初始化 CLS 扩展矩阵，全 0 ----
        with_cls_spatial_bias = torch.zeros(
            B, H, N + 1, N + 1, device=spatial_bias.device, dtype=spatial_bias.dtype
        )

        # ---- 节点间 bias 填回去 ----
        with_cls_spatial_bias[:, :, 1:, 1:] = spatial_bias

        return with_cls_spatial_bias


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
