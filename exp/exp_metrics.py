# coding : utf-8
# Author : yuxiang Zeng

import torch as t
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import stats


def ErrorMetrics(realVec, estiVec, config):
    """ 根据任务类型选择合适的误差计算方式 """
    if isinstance(realVec, np.ndarray):
        realVec = realVec.astype(float)
    elif isinstance(realVec, t.Tensor):
        realVec = realVec.cpu().detach().numpy().astype(float)

    if isinstance(estiVec, np.ndarray):
        estiVec = estiVec.astype(float)
    elif isinstance(estiVec, t.Tensor):
        estiVec = estiVec.cpu().detach().numpy().astype(float)

    if config.classification:
        return compute_classification_metrics(realVec, estiVec)
    else:
        return compute_regression_metrics(realVec, estiVec)



def dcg(scores):
    return np.sum((2 ** scores - 1) / np.log2(np.arange(2, len(scores) + 2)))


def ndcg_k(true_latency, pred_latency, k=None):
    """ NDCG@K，延迟越小越重要，得分越高 """
    if k is None:
        k = len(true_latency)

    true_latency = np.array(true_latency)
    pred_latency = np.array(pred_latency)

    # 替代负指数：构造正的 relevance 分数（越快越重要）
    true_relevance = np.max(true_latency) - true_latency

    pred_indices = np.argsort(pred_latency)[:k]
    ideal_indices = np.argsort(true_latency)[:k]

    dcg_score = dcg(true_relevance[pred_indices])
    idcg_score = dcg(true_relevance[ideal_indices])
    
    return dcg_score / idcg_score if idcg_score > 0 else 0.0


def compute_regression_metrics(realVec, estiVec):
    """计算回归任务 + 排序一致性的评估指标"""
    realVec = np.array(realVec).flatten()
    estiVec = np.array(estiVec).flatten()

    absError = np.abs(estiVec - realVec)

    MAE = np.mean(absError)
    MSE = np.mean((realVec - estiVec) ** 2)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs((realVec - estiVec) / realVec + 1e-8))  # 防止除0

    NMAE = np.sum(absError) / np.sum(np.abs(realVec))
    NRMSE = np.sqrt(np.sum((realVec - estiVec) ** 2)) / np.sqrt(np.sum(realVec ** 2))

    # Accuracy under thresholds
    thresholds = [0.01, 0.05, 0.10]
    Acc = [np.mean((absError < (realVec * t)).astype(float)) for t in thresholds]

    # 排序指标
    kendall_tau = stats.kendalltau(realVec, estiVec).correlation
    spearman_rho = stats.spearmanr(realVec, estiVec).correlation
    ndcg_score = ndcg_k(realVec, estiVec, 20)
    
    # 排序索引（越小越靠前）
    # real_sorted_idx = np.argsort(realVec)
    # esti_sorted_idx = np.argsort(estiVec)
    # topk = 20
    # print("【真实延迟排序前20的索引】")
    # print(real_sorted_idx[:topk].tolist())
    # print("【预测延迟排序前20的索引】")
    # print(esti_sorted_idx[:topk].tolist())
    # overlap = np.intersect1d(real_sorted_idx[:topk], esti_sorted_idx[:topk])
    # print(f"前{topk}中，预测命中真实Top-{topk}的数量: {len(overlap)}")
    # print(f"重叠索引: {overlap.tolist()}")
    
    return {
        'MAE': MAE,
        'MSE': MSE,
        'RMSE': RMSE,
        'MAPE': MAPE,
        'NMAE': NMAE,
        'NRMSE': NRMSE,
        'Acc_10': Acc[2],
        'KendallTau': kendall_tau,
        'SpearmanRho': spearman_rho,
        'NDCG': ndcg_score,
    }


def compute_classification_metrics(realVec, estiVec):
    """ 计算分类任务的指标 """
    AC = accuracy_score(realVec, estiVec)
    PR = precision_score(realVec, estiVec, average='macro', zero_division=0)
    RC = recall_score(realVec, estiVec, average='macro', zero_division=0)
    F1 = f1_score(realVec, estiVec, average='macro')

    return {
        'AC': AC,
        'PR': PR,
        'RC': RC,
        'F1': F1,
    }
