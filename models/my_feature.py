import torch
import torch.nn.functional as F
from collections import deque


def laplacian_node_ids_from_adj(
    adj: torch.Tensor, dp: int, drop_first: bool = True, unify_sign: bool = True
):
    """
    从邻接矩阵提取 Laplacian PE 特征向量。

    参数:
      adj: [N, N] 邻接矩阵 (0/1 或加权)
      dp:  需要的 LapPE 维度，返回 [N, dp]
      drop_first: 是否丢掉第一列常数特征向量 (特征值≈0)，默认 True
      unify_sign: 是否统一符号 (避免 ± 号不定)

    返回:
      P: [N, dp] 特征向量矩阵
    """
    N = adj.size(0)
    A = adj.float()
    deg = A.sum(dim=1).clamp(min=1e-12)
    d_inv_sqrt = deg.pow(-0.5)

    # 拉普拉斯矩阵
    L = (
        torch.eye(N, device=A.device, dtype=A.dtype)
        - d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    )
    L = 0.5 * (L + L.T)  # 数值对称化

    evals, evecs = torch.linalg.eigh(L)  # evals升序，evecs列与之配对

    if drop_first and evecs.size(1) > 0:
        evecs = evecs[:, 1:]  # 丢常数向量

    k = min(dp, evecs.size(1))
    P = evecs[:, :k] if k > 0 else evecs.new_zeros(N, 0)

    # 符号统一
    if unify_sign and P.numel() > 0:
        sign = torch.sign(P[0:1, :])
        sign[sign == 0] = 1.0
        P = P * sign

    # 如果 k < dp，右侧补零
    if k < dp:
        P = F.pad(P, (0, dp - k))

    return P  # [N, dp]


def in_out_degree_binary(adj: torch.Tensor, *, include_self_loops: bool = False):
    """
    0/1 有向邻接矩阵的入/出度（计边数）。
    返回: in_deg, out_deg  (均为 [n], dtype=int64)
    """
    assert adj.dim() == 2 and adj.size(0) == adj.size(1), "adj must be [n,n]"
    A = adj != 0  # 二值化，非零即边
    if not include_self_loops:
        idx = torch.arange(A.size(0), device=A.device)
        A = A.clone()
        A[idx, idx] = False
    out_deg = A.sum(dim=1).to(torch.int64)  # 行和
    in_deg = A.sum(dim=0).to(torch.int64)  # 列和
    return in_deg, out_deg


def apsp_unweighted_bfs(adj: torch.Tensor, *, directed: bool = True) -> torch.Tensor:
    INF_INT = 10**9
    """
    0/1 邻接矩阵的 APSP（所有点对最短路，单位权=1）。
    返回: D [n,n] (float32)，D[i,j]=跳数；不可达=+inf；对角=0.
    """
    assert adj.dim() == 2 and adj.size(0) == adj.size(1), "adj must be [n,n]"
    n = adj.size(0)
    A = adj != 0
    if not directed:
        A = A | A.T  # 无向化（如需）

    # 预构邻接表（Python list 更快）
    nbrs = [torch.where(A[i])[0].tolist() for i in range(n)]

    D_int = torch.full((n, n), INF_INT, dtype=torch.int32, device=adj.device)
    idx = torch.arange(n, device=adj.device)
    D_int[idx, idx] = 0

    for s in range(n):
        dist = D_int[s]
        q = deque([s])
        while q:
            v = q.popleft()
            dv = int(dist[v].item())
            for w in nbrs[v]:
                if dist[w] == INF_INT:
                    dist[w] = dv + 1
                    q.append(w)

    # 转成 float，并把不可达置为 +inf
    D = D_int.to(torch.float32)
    # D[D_int == INF_INT] = float('inf')
    # 2025年09月18日11:11:29
    D[D_int == INF_INT] = int(14)
    return D


def preprocess_binary_inout_and_spd(
    adj: torch.Tensor,
    *,
    include_self_loops_in_degree: bool = False,
    directed_for_spd: bool = True,
):
    """
    仅针对 0/1 有向邻接矩阵的输入侧预处理：
      - 二值入/出度（计边数）
      - 无权最短路矩阵（单位跳数），不可达=+inf
    """
    in_deg, out_deg = in_out_degree_binary(
        adj, include_self_loops=include_self_loops_in_degree
    )
    D = apsp_unweighted_bfs(adj, directed=directed_for_spd)
    return in_deg, out_deg, D


"""
    feats = preprocess_binary_inout_and_spd(
        adj01,
        include_self_loops_in_degree=False,
        directed_for_spd=True,   # 如需无向最短路可改成 False
    )
    print("in_deg :", feats["in_deg"])   # tensor([...], dtype=int64)
    print("out_deg:", feats["out_deg"])  # tensor([...], dtype=int64)
    print("D:\n", feats["D"])            # float32, +inf 表示不可达
    
"""
