# coding : utf-8
# Author : yuxiang Zeng

import torch as t
import numpy as np


def ErrorMetrics(realVec, estiVec, config):
    """根据任务类型选择合适的误差计算方式"""
    if isinstance(realVec, np.ndarray):
        realVec = realVec.astype(float)
    elif isinstance(realVec, t.Tensor):
        realVec = realVec.cpu().detach().numpy().astype(float)

    if isinstance(estiVec, np.ndarray):
        estiVec = estiVec.astype(float)
    elif isinstance(estiVec, t.Tensor):
        estiVec = estiVec.cpu().detach().numpy().astype(float)

    estiVec = estiVec.reshape(realVec.shape)

    if config.classification:
        return compute_classification_metrics(realVec, estiVec)
    else:
        return compute_regression_metrics(realVec, estiVec)


def compute_regression_metrics(realVec, estiVec):
    from scipy.stats import stats

    """计算回归任务 + 排序一致性的评估指标"""
    realVec = np.array(realVec).flatten()
    estiVec = np.array(estiVec).flatten()

    absError = np.abs(estiVec - realVec)

    MAE = np.mean(absError)
    MSE = np.mean((realVec - estiVec) ** 2)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs(realVec - estiVec) / (realVec + 1e-8))  # 防止除0

    NMAE = np.sum(absError) / np.sum(np.abs(realVec))
    NRMSE = np.sqrt(np.sum((realVec - estiVec) ** 2)) / np.sqrt(np.sum(realVec**2))

    # Accuracy under thresholds
    thresholds = [0.01, 0.05, 0.10]
    Acc = [np.mean((absError < (realVec * t)).astype(float)) for t in thresholds]

    # 排序指标
    kendall_tau = stats.kendalltau(realVec, estiVec).correlation
    spearman_rho = stats.spearmanr(realVec, estiVec).correlation

    return {
        "MAPE": MAPE,
        "KendallTau": kendall_tau,
        "Acc_10": Acc[2],
        "MAE": MAE,
        # "MSE": MSE,
        # "RMSE": RMSE,
        # "NMAE": NMAE,
        # "NRMSE": NRMSE,
        # "SpearmanRho": spearman_rho,
    }


def compute_classification_metrics(realVec, estiVec):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    """计算分类任务的指标"""
    AC = accuracy_score(realVec, estiVec)
    PR = precision_score(realVec, estiVec, average="macro", zero_division=0)
    RC = recall_score(realVec, estiVec, average="macro", zero_division=0)
    F1 = f1_score(realVec, estiVec, average="macro")

    return {
        "AC": AC,
        "PR": PR,
        "RC": RC,
        "F1": F1,
    }
