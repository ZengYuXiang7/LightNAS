
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
        self.op_embedding = DiscreteEncoder(num_operations=7, encoding_dim=self.d_model, encoding_type=config.op_encoder, output_dim=self.d_model)
        self.indeg_embedding = DiscreteEncoder(num_operations=10, encoding_dim=self.d_model, encoding_type='embedding', output_dim=self.d_model)
        self.outdeg_embedding = DiscreteEncoder(num_operations=10, encoding_dim=self.d_model, encoding_type='embedding', output_dim=self.d_model)
        
        self.att_bias = SPDSpatialBias(num_heads=config.num_heads, max_dist=99)    
        self.tokenGT = TokenGT(c_node=self.d_model, d_model=self.d_model, lap_dim=config.lp_d_model, use_edge=False)
        
        self.encoder = Transformer(self.d_model, config.num_layers, config.num_heads, 'rms', 'ffn', config.att_method)
        # enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        # self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        self.pred_head = nn.Linear(config.d_model, 1)  # 回归或分类
        
        
    def forward(self, graphs, features, eigvec, indgree, outdegree, dij):
        # seq_embeds = self.op_embedding(features)
        # print(indgree.shape, outdegree.shape, self.indeg_embedding(indgree).shape)
        # exit()
        seq_embeds = self.op_embedding(features) + self.indeg_embedding(indgree) + self.outdeg_embedding(outdegree)
        
        
        # [B, seq_len, d_model]
        seq_embeds, att_mask = self.tokenGT(graphs, seq_embeds, eigvec)   
        
        # 获得距离嵌入
        if self.config.att_bias:
            att_bias = self.att_bias(dij)                           # [B, H, N, N]
        else:
            att_bias = None
                 
        cls_out = self.encoder(seq_embeds, att_bias)[:, 0, :]   # [B, d_model]

        y = self.pred_head(cls_out)                           # 回归或分类
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
        with_cls_spatial_bias = torch.zeros(B, H, N+1, N+1, device=spatial_bias.device, dtype=spatial_bias.dtype)
        with_cls_spatial_bias[:, :, 1:, 1:] = spatial_bias
        
        return with_cls_spatial_bias
    

class TokenGT(nn.Module):
    def __init__(self, c_node: int, d_model: int = 128, lap_dim: int = 8, use_edge: bool = False, lap_node_id_sign_flip=True):
        super().__init__()
        self.use_edge = use_edge
        self.lap_dim = lap_dim
        self.lap_node_id_sign_flip = lap_node_id_sign_flip
        
        # 节点 token 投影
        # self.node_proj = nn.Linear(c_node + 2 * lap_dim, d_model)
        self.lap_encoder = nn.Linear(1 * lap_dim, d_model, bias=False)
        
        # 边 token 投影 (用常数 + LapPE(u)+LapPE(v))
        if use_edge:
            self.edge_proj = nn.Linear(1 + 2 * lap_dim, d_model)

        # type embedding (0=node, 1=edge)
        self.type_embed = nn.Embedding(2, d_model)

        # [graph] special token
        self.graph_tok = nn.Parameter(torch.randn(1, 1, d_model))
    
    
    def forward(self, adj: torch.Tensor, node_feats: torch.Tensor, eigvec: torch.Tensor, num_nodes=None):
        """
        adj:        [B, n, n]
        node_feats: [B, n, xdim]
        P:          [B, n, lap_dim]  (外部已算好 LapPE；如需内部计算，可自行打开注释)
        """
        B, n, _ = adj.shape
        device = adj.device
        # padded_index, node_mask_tok, edge_mask_tok, edge_num, T = self.build_padded_index_from_adj(adj, num_nodes, undirected=True)
        # node_mask = self.get_node_mask(num_nodes, adj.device)  # [B, max(n_node)]
        # lap_node_id = self.handle_eigvec(eigvec, node_mask, self.lap_node_id_sign_flip)
        # lap_index_embed = self.get_index_embed(lap_node_id, node_mask, padded_index)  # [B, T, 2Dl]
        node_tok = node_feats + self.lap_encoder(eigvec)   # [B, n, d]
        
        # print(lap_node_id.shape)
        # lap_index_embed = self.get_index_embed_all(lap_node_id, node_mask)  # [B, T, 2Dl]
        # print(lap_index_embed.shape)
        # node_tok = node_feats + self.lap_encoder(lap_index_embed)   # [B, n, d]
        
        tokens = [node_tok]  # 先放节点

        if self.use_edge:
            edge_tokens, pad_mask_edges = self.egde_forward(adj, eigvec)        # [B, max_m, d], [B, max_m]
            if edge_tokens.size(1) > 0:
                tokens.append(edge_tokens)
        else:
            pad_mask_edges = torch.zeros(B, 0, dtype=torch.bool, device=device)

        # 图级 token
        graph_tok = self.graph_tok.expand(B, -1, -1)                       # [B, 1, d]
        seq_embeds = torch.cat([graph_tok] + tokens, dim=1)                       # [B, 1+n(+max_m), d]

        # key padding mask：前 1+n 个位置（图级 + 节点）都为有效(False)；边用 pad_mask_edges
        pad_mask = torch.cat([
            torch.zeros(B, 1 + n, dtype=torch.bool, device=device),
            pad_mask_edges
        ], dim=1)                                                          # [B, 1+n(+max_m)]

        return seq_embeds, pad_mask
    
    
    def egde_forward(self, adj: torch.Tensor, P: torch.Tensor):
        """
        将 batch 内每个图的上三角边抽取为变长序列，并投影成边 token。
        输入:
            adj: [B, n, n] (bool/0-1/权重都可；会用 >0 变为 bool)
            P:   [B, n, lap_dim]
        返回:
            edge_tokens:   [B, max_m, d]（若无边则 [B,0,d]）
            pad_mask_edges:[B, max_m]，True 表示 padding（无效位置）
        """
        B, n, _ = adj.shape
        d = self.node_proj.out_features
        device = adj.device

        edge_tok_per_b = [None] * B
        edge_len = torch.zeros(B, dtype=torch.long, device=device)

        for b in range(B):
            Ab = (adj[b] > 0)
            Ab = Ab.clone()
            Ab.fill_diagonal_(False)
            # 仅取上三角，避免重复
            src, dst = torch.triu(Ab, diagonal=1).nonzero(as_tuple=True)
            m = src.numel()
            edge_len[b] = m
            if m == 0:
                continue

            Pu, Pv = P[b, src], P[b, dst]                     # [m, lap], [m, lap]
            edge_input = torch.cat(
                [torch.ones(m, 1, device=device), Pu, Pv],    # [m, 1 + 2*lap]
                dim=-1
            )
            etok = self.edge_proj(edge_input).unsqueeze(0)     # [1, m, d]
            etok = etok + self.type_embed(
                torch.ones(m, dtype=torch.long, device=device)
            ).unsqueeze(0)                                     # 边类型=1
            edge_tok_per_b[b] = etok

        max_m = int(edge_len.max().item() if edge_len.numel() else 0)

        if max_m > 0:
            edge_tokens = torch.zeros(B, max_m, d, device=device)
            pad_mask_edges = torch.ones(B, max_m, dtype=torch.bool, device=device)  # True=pad
            for b, etok in enumerate(edge_tok_per_b):
                if etok is None:
                    continue
                L = etok.size(1)
                edge_tokens[b, :L] = etok[0]
                pad_mask_edges[b, :L] = False                 # False=有效
        else:
            edge_tokens = torch.zeros(B, 0, d, device=device)
            pad_mask_edges = torch.zeros(B, 0, dtype=torch.bool, device=device)

        return edge_tokens, pad_mask_edges
    
    
    def build_padded_index_from_adj(
        self, adj: torch.Tensor, 
        num_nodes: torch.Tensor,
        undirected: bool = True
    ):
        """
        根据 batched 邻接矩阵与每图节点数，构造混合序列的 padded_index 与掩码。

        输入:
        adj:       [B, N, N] 邻接矩阵（0/1 或加权；可含 padding 节点行列）
        num_nodes: [B]       每张图的真实节点数（int/long）
        undirected:          若为 True，则仅取上三角(去重)；否则取所有非零有向边

        返回:
        padded_index:   [B, T, 2]，第 b 个图第 t 个 token 的节点对索引 (u,v)
                        - 前 n_b 个位置是节点 token： (i,i), i=0..n_b-1
                        - 后 m_b 个位置是边 token：   (u,v)
                        - 其余位置为 0（padding）
        padded_node_mask: [B, T]，True 表示节点 token 位置
        padded_edge_mask: [B, T]，True 表示边 token 位置
        edge_num:         [B]，每图的边数 m_b（按上面抽取规则）
        """
        device = adj.device
        B, N, _ = adj.shape
        num_nodes = num_nodes.to(device).long()

        # 逐图统计边，确定每图序列长度 n_b + m_b，再 pad 到批内最大 T
        edge_lists = []
        edge_num = torch.zeros(B, dtype=torch.long, device=device)
        seq_len = torch.zeros(B, dtype=torch.long, device=device)

        for b in range(B):
            n_b = int(num_nodes[b].item())
            if n_b == 0:
                edge_lists.append((None, None))  # 无节点
                edge_num[b] = 0
                seq_len[b]  = 0
                continue

            Ab = (adj[b, :n_b, :n_b] > 0).clone()
            Ab.fill_diagonal_(False)

            if undirected:
                src, dst = torch.triu(Ab, diagonal=1).nonzero(as_tuple=True)  # 去重
            else:
                src, dst = Ab.nonzero(as_tuple=True)                           # 有向

            m_b = int(src.numel())
            edge_lists.append((src, dst))
            edge_num[b] = m_b
            seq_len[b]  = n_b + m_b

        # 统一序列长度
        T = int(seq_len.max().item()) if B > 0 else 0

        # 构造输出张量
        padded_index = torch.zeros(B, T, 2, dtype=torch.long, device=device)
        padded_node_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        padded_edge_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

        for b in range(B):
            n_b = int(num_nodes[b].item())
            m_b = int(edge_num[b].item())

            if n_b > 0:
                # 节点 token: (i,i)
                idx = torch.arange(n_b, device=device, dtype=torch.long)
                padded_index[b, :n_b, 0] = idx
                padded_index[b, :n_b, 1] = idx
                padded_node_mask[b, :n_b] = True

            if m_b > 0:
                src, dst = edge_lists[b]
                padded_index[b, n_b:n_b+m_b, 0] = src
                padded_index[b, n_b:n_b+m_b, 1] = dst
                padded_edge_mask[b, n_b:n_b+m_b] = True

        return padded_index, padded_node_mask, padded_edge_mask, edge_num, T
    
    @staticmethod
    @torch.no_grad()
    def get_node_mask(node_num, device):
        b = len(node_num)
        max_n = max(node_num)
        node_index = torch.arange(max_n, device=device, dtype=torch.long)[None, :].expand(b, max_n)  # [B, max_n]
        node_num = torch.tensor(node_num, device=device, dtype=torch.long)[:, None]  # [B, 1]
        node_mask = torch.less(node_index, node_num)  # [B, max_n]
        return node_mask
    
    @staticmethod
    @torch.no_grad()
    def get_random_sign_flip(eigvec, node_mask):
        b, max_n = node_mask.size()
        d = eigvec.size(1)

        sign_flip = torch.rand(b, d, device=eigvec.device, dtype=eigvec.dtype)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        sign_flip = sign_flip[:, None, :].expand(b, max_n, d)
        sign_flip = sign_flip[node_mask]
        return sign_flip

    def handle_eigvec(self, eigvec, node_mask, sign_flip):
        if sign_flip and self.training:
            sign_flip = self.get_random_sign_flip(eigvec, node_mask)
            eigvec = eigvec * sign_flip
        else:
            pass
        return eigvec
    
    
    @staticmethod
    def get_index_embed(node_id, node_mask, padded_index):
        """
        :param node_id: Tensor([sum(node_num), D])
        :param node_mask: BoolTensor([B, max_n])
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, 2D])
        """
        b, max_n = node_mask.size()
        max_len = padded_index.size(1)
        d = node_id.size(-1)

        padded_node_id = torch.zeros(b, max_n, d, device=node_id.device, dtype=node_id.dtype)  # [B, max_n, D]
        padded_node_id[node_mask] = node_id

        padded_node_id = padded_node_id[:, :, None, :].expand(b, max_n, 2, d)
        padded_index = padded_index[..., None].expand(b, max_len, 2, d)
        
        index_embed = padded_node_id.gather(1, padded_index)  # [B, T, 2, D]
        
        index_embed = index_embed.view(b, max_len, 2 * d)
        
        return index_embed
    
    
    def get_padded_index(self, node_feature, edge_index, node_num, edge_num):  
        seq_len = [n + e for n, e in zip(node_num, edge_num)]
        b = len(seq_len)
        d = node_feature.size(-1)
        max_len = max(seq_len)
        max_n = max(node_num)
        device = edge_index.device

        token_pos = torch.arange(max_len, device=device)[None, :].expand(b, max_len)  # [B, T]

        seq_len = torch.tensor(seq_len, device=device, dtype=torch.long)[:, None]  # [B, 1]
        node_num = torch.tensor(node_num, device=device, dtype=torch.long)[:, None]  # [B, 1]
        edge_num = torch.tensor(edge_num, device=device, dtype=torch.long)[:, None]  # [B, 1]

        node_index = torch.arange(max_n, device=device, dtype=torch.long)[None, :].expand(b, max_n)  # [B, max_n]
        node_index = node_index[None, node_index < node_num].repeat(2, 1)  # [2, sum(node_num)]

        padded_node_mask = torch.less(token_pos, node_num)
        padded_edge_mask = torch.logical_and(
            torch.greater_equal(token_pos, node_num),
            torch.less(token_pos, node_num + edge_num)
        )

        padded_index = torch.zeros(b, max_len, 2, device=device, dtype=torch.long)  # [B, T, 2]
        padded_index[padded_node_mask, :] = node_index.t()
        padded_index[padded_edge_mask, :] = edge_index.t()

        return padded_index, padded_node_mask, padded_edge_mask



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
    

class SoftRankLoss(torch.nn.Module):
    def __init__(self, config):
        super(SoftRankLoss, self).__init__()
        self.config = config
        if config.try_exp == 1:
            self.sp_tau = 1
            self.kd_alpha = 10
            self.kd_jitter = 0
        elif config.try_exp == 2:
            self.sp_tau = 3
            self.kd_alpha = 10
            self.kd_jitter = 0
        elif config.try_exp == 3:
            self.sp_tau = 1
            self.kd_alpha = 3
            self.kd_jitter = 0  
        
    # Kendall
    def diffkendall_tau(self, x, y, alpha=10.0, jitter=0.0):
        x = x.view(-1)
        y = y.view(-1)
        if jitter > 0:
            x = x + jitter * torch.randn_like(x)
            y = y + jitter * torch.randn_like(y)
        n = x.numel()

        dx = x.unsqueeze(0) - x.unsqueeze(1)   # [n,n]
        dy = y.unsqueeze(0) - y.unsqueeze(1)

        sx = torch.tanh(alpha * dx)
        sy = torch.tanh(alpha * dy)

        mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        concord = (sx * sy)[mask].sum()

        num_pairs = torch.tensor(n * (n - 1) / 2, device=x.device, dtype=concord.dtype)
        return concord / num_pairs

    def diffkendall_loss(self, pred, target, alpha=10.0, jitter=0.0):
        return 1.0 - self.diffkendall_tau(pred, target, alpha, jitter=jitter)

    
    # Spearman
    def soft_rank(self, v, tau=1.0):
        v = v.view(-1, 1)
        P = torch.sigmoid((v.T - v) / tau)     # [n,n]
        r = 1.0 + P.sum(dim=1)                 # 近似秩
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

        loss_spearman = self.spearman_loss(preds, labels, tau=self.sp_tau)
        loss_kendall = self.diffkendall_loss(preds, labels, alpha=self.kd_alpha, jitter=self.kd_jitter)

        return loss_spearman, loss_kendall
    
    