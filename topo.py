import numpy as np
import networkx as nx

def build_node_topo_features(
    A: np.ndarray,
    use_degree=True,
    use_core=True,
    use_tri_cluster=True,
    use_centrality=True,          # PR/Closeness/Harmonic/Eigenvector（相对便宜）
    use_rwpe=True, K_rw=16,       # Random Walk PE（对角访问概率）
    use_lpe=True, d_pe=16,        # Laplacian PE（前 d_pe 特征向量，含符号对齐）
    use_anchor=True, R_anchor=16, Dmax=8,  # Anchor 距离分桶 + 平均
    normalize="zscore",           # None | "zscore"
    seed=0
):
    """
    输入:
      A: 邻接矩阵 (n,n), numpy
    输出:
      X: (n, dim) 节点拓扑特征拼接矩阵（不含任何投影/MLP）
      columns: list[ (name, start_idx, end_idx) ] 便于追踪每段特征的位置
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A 必须是方阵"
    n = A.shape[0]
    G = nx.from_numpy_array(A)  # 无向无权；如需有向/权重可改这里

    feats = []
    columns = []
    col = 0

    def add_feat(name, arr_2d):
        nonlocal feats, columns, col
        assert arr_2d.shape[0] == n
        d = arr_2d.shape[1]
        feats.append(arr_2d)
        columns.append((name, col, col + d))
        col += d

    # --------- 基础标量 ---------
    if use_degree:
        deg = np.array([d for _, d in G.degree()], float)[:, None]
        add_feat("deg", deg)
        add_feat("log1p_deg", np.log1p(deg))

    if use_core:
        kcore = np.array([c for _, c in nx.core_number(G).items()], float)[:, None]
        add_feat("kcore", kcore)

    if use_tri_cluster:
        tri = np.array([t for _, t in nx.triangles(G).items()], float)[:, None]
        clus = np.array([c for _, c in nx.clustering(G).items()], float)[:, None]
        add_feat("triangles", tri)
        add_feat("clustering", clus)

    # --------- 中心性（相对便宜的几种）---------
    if use_centrality:
        pr = np.array(list(nx.pagerank(G, alpha=0.85).values()), float)[:, None]
        close = np.array(list(nx.closeness_centrality(G).values()), float)[:, None]
        harm = np.array(list(nx.harmonic_centrality(G).values()), float)[:, None]
        eig = np.array(list(nx.eigenvector_centrality_numpy(G).values()), float)[:, None]
        add_feat("pagerank", pr)
        add_feat("closeness", close)
        add_feat("harmonic", harm)
        add_feat("eigenvector", eig)

    # --------- RWPE（随机游走对角特征）---------
    if use_rwpe and n > 0:
        P = A.astype(float)
        row_sum = P.sum(1, keepdims=True) + 1e-9
        P = P / row_sum
        diag_feats = []
        Pk = np.eye(n)
        for _ in range(1, K_rw + 1):
            Pk = Pk @ P
            diag_feats.append(np.diag(Pk))
        rwpe = np.stack(diag_feats, axis=1)  # (n, K_rw)
        add_feat(f"rwpe_K{K_rw}", rwpe)

    # --------- LPE（拉普拉斯特征向量 + 符号对齐）---------
    if use_lpe and n > 0:
        L = nx.laplacian_matrix(G).toarray()
        vals, vecs = np.linalg.eigh(L)  # vecs: (n, n) 列为特征向量
        idx = np.argsort(vals)[: min(d_pe, n)]
        U = vecs[:, idx]                # (n, d_pe)
        # 符号对齐：与度做相关性，为负则翻转
        deg_vec = np.array([d for _, d in G.degree()], float)[:, None]
        centered = deg_vec - deg_vec.mean()
        s = np.sign((U * centered).sum(0) + 1e-12)
        U = U * s  # (n, d_pe)
        add_feat(f"lpe_d{U.shape[1]}", U)

    # --------- Anchor 距离编码（分桶 + 平均）---------
    if use_anchor and n > 0:
        rng = np.random.default_rng(seed)
        anchors = sorted(rng.choice(n, size=min(R_anchor, n), replace=False).tolist())
        spd = dict(nx.all_pairs_shortest_path_length(G))
        D_anchor = np.zeros((n, len(anchors)), float)
        INF = 10**6
        for i in range(n):
            for j, a in enumerate(anchors):
                D_anchor[i, j] = spd[i].get(a, INF)
        # 分桶：0..Dmax，>Dmax→Dmax+1
        Db = np.clip(D_anchor, 0, Dmax + 1).astype(int)  # (n, R)
        # one-hot 后对 R 取平均，得到 (n, Dmax+2)
        oh = np.eye(Dmax + 2, dtype=float)[Db]  # (n,R,Dmax+2)
        anchor_enc = oh.mean(1)                 # (n,Dmax+2)
        add_feat(f"anchor_buckets_R{len(anchors)}_D{Dmax}", anchor_enc)

    # --------- 拼接 & 归一化 ---------
    if len(feats) == 0:
        X = np.zeros((n, 0), dtype=float)
        columns = []
    else:
        X = np.concatenate(feats, axis=1)

    if normalize == "zscore" and X.size > 0:
        mu = X.mean(0, keepdims=True)
        sigma = X.std(0, keepdims=True) + 1e-6
        X = (X - mu) / sigma

    return X, columns


# ============ 使用示例 ============
if __name__ == "__main__":
    A = np.array([
        [0,1,1,0,0],
        [1,0,1,0,0],
        [1,1,0,1,0],
        [0,0,1,0,1],
        [0,0,0,1,0],
    ], dtype=float)

    X, columns = build_node_topo_features(
        A,
        use_degree=True,
        use_core=True,
        use_tri_cluster=True,
        use_centrality=True,
        use_rwpe=True, K_rw=8,
        use_lpe=True, d_pe=8,
        use_anchor=True, R_anchor=8, Dmax=6,
        normalize="zscore",
        seed=42
    )
    print("X shape:", X.shape)           # (n, dim)
    print("列段信息:")
    for name, s, e in columns:
        print(f"  {name:28s} -> [{s}, {e}) 维数={e-s}")