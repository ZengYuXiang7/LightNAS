
# 请使用python 3.10版本运行此代码
from data_process.create_latency import *
from tqdm import *
import networkx as nx
import pickle 
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from karateclub import Graph2Vec


def get_dataset_graph(dataset='201_acc'):
    raw_data = pickle.load(open('data/201_acc_data.pkl', 'rb'))
    all_keys = raw_data['key']
    
    all_graphs = []
    if dataset == '201_acc':
        for key in tqdm(all_keys):
            arch_str = get_arch_str_from_arch_vector(key)             # 架构向量转字符串
            adj_matrix, features = info2mat(arch_str) 
            graph = nx.from_numpy_array(adj_matrix)   # 邻接矩阵 -> 图
            # 添加节点特征
            for j in range(graph.number_of_nodes()):
                graph.nodes[j]['feature'] = np.eye(7, dtype=int)[features[j]]
                # graph.nodes[j]['feature'] = features[j]
            all_graphs.append(graph)
    elif dataset == 'nasbench101':
        pass 
    
    elif dataset == 'nnlqp':
        pass
    
    
    return all_graphs


def get_graph2vec_clusters(graphs, dimensions=64, epochs=10):
    """
    使用 Graph2Vec 生成图嵌入，并进行 KMeans 聚类。

    Parameters
    ----------
    graphs : list of networkx.Graph
        输入的一组图
    dimensions : int
        嵌入维度
    workers : int
        并行 worker 数
    epochs : int
        训练轮数
    n_clusters : int
        聚类簇数
    random_state : int
        随机种子

    Returns
    -------
    embeddings : np.ndarray
        图的嵌入表示
    labels : np.ndarray
        每个图的聚类标签
    """
    # Graph2Vec 表征
    model = Graph2Vec(dimensions=dimensions, workers=4, epochs=epochs)
    model.fit(graphs)
    embeddings = model.get_embedding()

    return embeddings



def select_diverse_by_kmeans(embeddings: np.ndarray, n_clusters: int = 100):
    # 归一化能让距离更稳定（特别是用余弦度量时）
    X = normalize(embeddings)  # L2-normalize
    km = KMeans(n_clusters=n_clusters, n_init="auto")
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    # 每簇挑“离簇中心最近”的索引
    selected_idx = []
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        # 计算该簇内到中心的距离
        dist = np.linalg.norm(X[idx] - centers[c], axis=1)
        rep = idx[np.argmin(dist)]
        selected_idx.append(rep)

    return np.array(selected_idx, dtype=int)


# sample_idx = sample_method('ours', sample_num=100)
def sample_method(method='ours', sample_num=100):
    if method == 'random':
        cluster_idx = np.random.choice(len(all_graphs), sample_num, replace=False)
    elif method == 'ours':
        all_graphs = get_dataset_graph(dataset='201_acc')
        embeddings = get_graph2vec_clusters(all_graphs, dimensions=32, epochs=10)
        cluster_idx = select_diverse_by_kmeans(embeddings, n_clusters=sample_num)
    return cluster_idx


if __name__ == "__main__":
    with open('./data/201_traing_sample.pkl', 'wb') as f:
        all_cluster_idx = {}
        all_sanerios = [152, 456, 764, 1528]
        for sanerio in all_sanerios:
            print(f"=== Sanerio {sanerio} ===")
            cluster_idx = sample_method('ours', sample_num=sanerio)
            print(len(cluster_idx), cluster_idx[:10])
            all_cluster_idx[sanerio] = cluster_idx
        pickle.dump(all_cluster_idx, f)
    print('Done!')