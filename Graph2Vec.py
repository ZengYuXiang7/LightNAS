# coding : utf-8
# Author : Yang Wang

import numpy as np
import networkx as nx
from karateclub import Graph2Vec
from karateclub import FeatherGraph
from sklearn.cluster import KMeans

def graph_clustering(adjs, features, model_type='Graph2Vec', dimensions=64, workers=2, epochs=10, n_clusters=3, random_state=42):
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
    if model_type == 'Graph2Vec':
        # 如果选择Graph2Vec，调用graph_clustering_Graph2Vec
        return graph_clustering_Graph2Vec(adjs, features, dimensions=dimensions, workers=workers, epochs=epochs, 
                                          n_clusters=n_clusters, random_state=random_state)
    
    elif model_type == 'FeatherGraph':
        # 如果选择FeatherGraph，调用graph_clustering_FeatherGraph
        return graph_clustering_FeatherGraph(adjs, features, n_clusters=n_clusters, dimensions=dimensions, 
                                             epochs=epochs, random_state=random_state)
    
    else:
        raise ValueError("模型类型无效!")


def graph_clustering_Graph2Vec(adjs, features, dimensions=64, workers=2, epochs=10, 
                     n_clusters=3, random_state=42):
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
        G = nx.from_numpy_array(adj)   # 邻接矩阵 -> 图
        # 添加节点特征
        for j in range(n):
            G.nodes[j]['feature'] = features[i][j].tolist()
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

def graph_clustering_FeatherGraph(adjs, features, n_clusters=3, dimensions=64, epochs=10, random_state=42):
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
        G = nx.from_numpy_array(adj)   # 邻接矩阵 -> 图
        # 添加节点特征
        for j in range(n):
            G.nodes[j]['feature'] = features[i][j].tolist()
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


# 测试代码
m, n, d = 15000, 20, 5  # 10 个图，每个图 20 节点，每个节点 5 维特征
adjs = np.random.randint(0, 2, size=(m, n, n))  # 随机生成邻接矩阵
features = np.random.randn(m, n, d)  # 随机生成节点特征

# 调用函数获取聚类标签
labels = graph_clustering(adjs, features, model_type='Graph2Vec', dimensions=128, workers=4, epochs=20, n_clusters=3, random_state=42)

# 输出聚类标签
print("聚类标签:", labels)

