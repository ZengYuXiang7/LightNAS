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

# coding : utf-8
# Author : Yang Wang

import numpy as np
import networkx as nx
from karateclub import Graph2Vec
from karateclub import FeatherGraph
from sklearn.cluster import KMeans


def graph_clustering(
    adjs,
    features,
    model_type="Graph2Vec",
    dimensions=64,
    workers=2,
    epochs=10,
    n_clusters=3,
    random_state=42,
):
    """
    根据模型类型执行不同的图聚类方法。
    :param adjs: 邻接矩阵, 形状为 (m, n, n), m是图的数量，n是每个图的节点数
    :param features: 节点特征, 形状为 (m, n, d), m是图的数量，n是节点数，d是节点特征的维度
    :param model_type: 使用的模型类型 ('Graph2Vec' 或 'FeatherGraph')
    :param dimensions: 嵌入维度
    :param workers: Graph2Vec 的工作线程数
    :param epochs: 训练轮数
    :param n_clusters: KMeans 的聚类簇数
    :param random_state: KMeans 的随机种子
    :return: 聚类标签 (kmeans.labels_)
    """
    if model_type == "Graph2Vec":
        # 如果选择Graph2Vec，调用graph_clustering_Graph2Vec
        return graph_clustering_Graph2Vec(
            adjs,
            features,
            dimensions=dimensions,
            workers=workers,
            epochs=epochs,
            n_clusters=n_clusters,
            random_state=random_state,
        )

    elif model_type == "FeatherGraph":
        # 如果选择FeatherGraph，调用graph_clustering_FeatherGraph
        return graph_clustering_FeatherGraph(
            adjs,
            features,
            n_clusters=n_clusters,
            dimensions=dimensions,
            epochs=epochs,
            random_state=random_state,
        )

    else:
        raise ValueError("模型类型无效!")


def graph_clustering_Graph2Vec(
    adjs, features, dimensions=64, workers=2, epochs=10, n_clusters=3, random_state=42
):
    """
    对图进行聚类
    :param adjs: 邻接矩阵, 形状为 (m, n, n), m是图的数量，n是每个图的节点数
    :param features: 节点特征, 形状为 (m, n, d), m是图的数量，n是节点数，d是节点特征的维度
    :param dimensions: Graph2Vec 嵌入维度
    :param workers: Graph2Vec 的工作线程数
    :param epochs: Graph2Vec 的训练轮数
    :param n_clusters: KMeans 的聚类簇数
    :param random_state: KMeans 的随机种子
    :return: 聚类标签 (kmeans.labels_)
    """

    # 构造图
    m, n, d = adjs.shape
    graphs = []
    for i in range(m):
        adj = adjs[i]
        G = nx.from_numpy_array(adj)  # 邻接矩阵 -> 图
        # 添加节点特征
        for j in range(n):
            G.nodes[j]["feature"] = features[i][j].tolist()
        graphs.append(G)

    # Graph2Vec 表征
    graph2vec = Graph2Vec(dimensions=dimensions, workers=workers, epochs=epochs)
    graph2vec.fit(graphs)

    embeddings = graph2vec.get_embedding()

    # KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embeddings)

    # 返回聚类标签
    return kmeans.labels_


def graph_clustering_FeatherGraph(
    adjs, features, n_clusters=3, dimensions=64, epochs=10, random_state=42
):
    """
    对图进行聚类
    :param adjs: 邻接矩阵, 形状为 (m, n, n), m是图的数量，n是每个图的节点数
    :param features: 节点特征, 形状为 (m, n, d), m是图的数量，n是节点数，d是节点特征的维度
    :param n_clusters: KMeans 的聚类簇数
    :param dimensions: FeatherGraph 嵌入维度
    :param epochs: FeatherGraph 的训练轮数
    :param random_state: KMeans 的随机种子
    :return: 聚类标签 (kmeans.labels_)
    """

    # 构造图
    m, n, d = adjs.shape
    graphs = []
    for i in range(m):
        adj = adjs[i]
        G = nx.from_numpy_array(adj)  # 邻接矩阵 -> 图
        # 添加节点特征
        for j in range(n):
            G.nodes[j]["feature"] = features[i][j].tolist()
        graphs.append(G)

    # FeatherGraph 表征
    feather_graph = FeatherGraph()
    feather_graph.fit(graphs)  # 直接传入多个图的列表

    embeddings = feather_graph.get_embedding()

    # KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, verbose=True)
    kmeans.fit(embeddings)

    # 返回聚类标签
    return kmeans.labels_


def get_dataset_graph(dataset="201_acc"):
    all_graphs = []
    if dataset == "201_acc":
        raw_data = pickle.load(open("data/201_acc_data.pkl", "rb"))
        all_keys = raw_data["key"]
        for key in tqdm(all_keys):
            arch_str = get_arch_str_from_arch_vector(key)  # 架构向量转字符串
            adj_matrix, features = info2mat(arch_str)
            graph = nx.from_numpy_array(adj_matrix)  # 邻接矩阵 -> 图
            # 添加节点特征
            for j in range(graph.number_of_nodes()):
                graph.nodes[j]["feature"] = np.eye(7, dtype=int)[features[j]]
                # graph.nodes[j]['feature'] = features[j]
            all_graphs.append(graph)

    elif dataset == "101_acc":
        raw_data = pickle.load(open("data/101_acc_data.pkl", "rb"))
        adj_matrix = raw_data["adj_matrix"]
        all_features = raw_data["features"]

        for i in tqdm(range(len(adj_matrix))):
            graph = nx.from_numpy_array(np.array(adj_matrix[i]))
            # 添加节点特征
            for j in range(graph.number_of_nodes()):
                node_id = np.array(all_features[i][j])
                # 注意： 这里检查一下是最大操作数是多少，几个操作元
                graph.nodes[j]["feature"] = np.eye(5, dtype=int)[node_id]

            all_graphs.append(graph)

    elif dataset == "nnlqp":
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
    # print(graphs)
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
def sample_method(method="ours", sample_num=100, dataset="201_acc"):
    if method == "random":
        cluster_idx = np.random.choice(len(all_graphs), sample_num, replace=False)
    elif method == "ours":
        all_graphs = get_dataset_graph(dataset=dataset)
        embeddings = get_graph2vec_clusters(all_graphs, dimensions=32, epochs=10)
        cluster_idx = select_diverse_by_kmeans(embeddings, n_clusters=sample_num)
    return cluster_idx


def f():
    # 测试代码
    m, n, d = 15000, 20, 5  # 10 个图，每个图 20 节点，每个节点 5 维特征
    adjs = np.random.randint(0, 2, size=(m, n, n))  # 随机生成邻接矩阵
    features = np.random.randn(m, n, d)  # 随机生成节点特征

    # 调用函数获取聚类标签
    labels = graph_clustering(
        adjs,
        features,
        model_type="Graph2Vec",
        dimensions=128,
        workers=4,
        epochs=20,
        n_clusters=3,
        random_state=42,
    )

    # 输出聚类标签
    print("聚类标签:", labels)
    return True


if __name__ == "__main__":

    with open("./data/201_traing_sample.pkl", "wb") as f:
        all_cluster_idx = {}
        all_sanerios = [152, 458, 764, 1528]
        for sanerio in all_sanerios:
            print(f"=== Sanerio {sanerio} ===")
            cluster_idx = sample_method("ours", sample_num=sanerio, dataset="201_acc")
            print(len(cluster_idx), cluster_idx[:10])
            all_cluster_idx[sanerio] = cluster_idx
        pickle.dump(all_cluster_idx, f)
    print("Done!")

    with open("./data/101_traing_sample.pkl", "wb") as f:
        all_cluster_idx = {}
        # 这里改一下
        all_sanerios = [102, 169, 423, 4236]
        for sanerio in all_sanerios:
            print(f"=== Sanerio {sanerio} ===")
            cluster_idx = sample_method("ours", sample_num=sanerio, dataset="101_acc")
            print(len(cluster_idx), cluster_idx[:10])
            all_cluster_idx[sanerio] = cluster_idx
        pickle.dump(all_cluster_idx, f)
    print("Done!")
