import torch
import torch.nn as nn
import torch.nn.functional as F


def laplacian_node_ids_from_adj(adj: torch.Tensor, dp: int) -> torch.Tensor:
    """adj: [B,n,n] → 返回 LapPE [B,n,dp]"""
    B, n, _ = adj.shape
    outs = []
    for b in range(B):
        A = adj[b].float()
        deg = A.sum(1).clamp(min=1e-12)
        D_inv_sqrt = torch.diag(deg.pow(-0.5))
        L = torch.eye(n, device=adj.device) - D_inv_sqrt @ A @ D_inv_sqrt
        _, evecs = torch.linalg.eigh(L)
        k = min(dp, n)
        P = evecs[:, :k]
        outs.append(F.pad(P, (0, dp - k)))
    return torch.stack(outs, dim=0)  # [B,n,dp]


class TokenGT(nn.Module):
    def __init__(self, c_node: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4,
                 lap_dim: int = 8, use_edge: bool = False):
        super().__init__()
        self.use_edge = use_edge
        self.lap_dim = lap_dim

        # 节点 token 投影
        self.node_proj = nn.Linear(c_node + lap_dim, d_model)

        # 边 token 投影 (用常数 + LapPE(u)+LapPE(v))
        if use_edge:
            self.edge_proj = nn.Linear(1 + 2 * lap_dim, d_model)

        # type embedding (0=node, 1=edge)
        self.type_embed = nn.Embedding(2, d_model)

        # [graph] special token
        self.graph_tok = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer 编码器
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, adj: torch.Tensor, node_feats: torch.Tensor):
        """
        adj: [B,n,n]
        node_feats: [B,n,c_node]
        """
        B, n, _ = adj.shape

        # Laplacian PE
        P = laplacian_node_ids_from_adj(adj, self.lap_dim)  # [B,n,lap_dim]

        # --- 节点 token ---
        node_tok = self.node_proj(torch.cat([node_feats, P], dim=-1))  # [B,n,d_model]
        node_tok = node_tok + self.type_embed(torch.zeros(n, dtype=torch.long, device=adj.device)).unsqueeze(0)

        tokens = [node_tok]

        # --- 边 token (可选) ---
        if self.use_edge:
            edge_tokens = []
            for b in range(B):
                src, dst = (adj[b] > 0).nonzero(as_tuple=True)  # 单张图的边索引
                m = src.size(0)  # 这张图的边数
                if m == 0:
                    continue
                edge_feat = torch.ones(m, 1, device=adj.device)
                Pu, Pv = P[b, src], P[b, dst]  # LapPE(u), LapPE(v)
                edge_input = torch.cat([edge_feat, Pu, Pv], dim=-1)  # [m, 1+2*lap_dim]
                etok = self.edge_proj(edge_input).unsqueeze(0)       # [1,m,d_model]
                # 加上 type embedding (1 = edge)
                etok = etok + self.type_embed(
                    torch.ones(m, dtype=torch.long, device=adj.device)
                ).unsqueeze(0)
                edge_tokens.append(etok)

            if edge_tokens:
                # pad 到相同长度
                max_m = max([e.size(1) for e in edge_tokens])
                padded = torch.zeros(B, max_m, self.node_proj.out_features, device=adj.device)
                pad_mask_edges = torch.ones(B, max_m, dtype=torch.bool, device=adj.device)
                for b, etok in enumerate(edge_tokens):
                    L = etok.size(1)
                    padded[b, :L] = etok
                    pad_mask_edges[b, :L] = False
                tokens.append(padded)

        # --- special token + 拼接 ---
        graph_tok = self.graph_tok.expand(B, -1, -1)  # [B,1,d_model]
        seq = torch.cat([graph_tok] + tokens, dim=1)  # [B,L,d_model]

        # --- Transformer 编码 ---
        pad_mask = torch.zeros(B, seq.size(1), dtype=torch.bool, device=adj.device)
        x_enc = self.encoder(seq, src_key_padding_mask=pad_mask)
        graph_repr = x_enc[:, 0, :]  # [graph] token 表示
        return graph_repr, x_enc


# ----------------------
# Demo
# ----------------------
if __name__ == "__main__":
    B, n, dim = 32, 6, 8  # 先用 B=1 测试
    adj = torch.randint(0, 2, (B, n, n))
    adj = torch.triu(adj, 1)
    adj = adj + adj.transpose(-1, -2)

    node_feats = torch.randn(B, n, dim)

    model = TokenGT(c_node=dim, d_model=128, nhead=8, num_layers=2, lap_dim=8, use_edge=False)  # use_edge=True 可生成边 token
    graph_repr, x_enc = model(adj, node_feats)
    print("graph_repr:", graph_repr.shape)  # [B,128]
    print("x_enc:", x_enc.shape)            # [B,L,128]
    
    
    model = TokenGT(c_node=dim, d_model=128, nhead=8, num_layers=2, lap_dim=8, use_edge=True)  # use_edge=True 可生成边 token
    graph_repr, x_enc = model(adj, node_feats)
    print("graph_repr:", graph_repr.shape)  # [B,128]
    print("x_enc:", x_enc.shape)            # [B,L,128]