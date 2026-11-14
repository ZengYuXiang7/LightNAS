"""延迟预测的排序与相对误差指标。"""

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr


def ErrorMetrics(realVec, estiVec):
    # 张量先还原为一维数组，统一神经网络和机器学习的指标计算。
    if isinstance(realVec, torch.Tensor):
        realVec = realVec.detach().cpu().numpy()
    if isinstance(estiVec, torch.Tensor):
        estiVec = estiVec.detach().cpu().numpy()
    real = np.asarray(realVec, dtype=float).reshape(-1)
    predicted = np.asarray(estiVec, dtype=float).reshape(-1)

    # MAPE 和 Acc_10 均以比例表示；NMAE、NRMSE 保持原有定义。
    absolute_error = np.abs(predicted - real)
    with np.errstate(divide='raise', invalid='raise'):
        relative_error = absolute_error / np.abs(real)
        nmae = np.sum(absolute_error) / np.sum(real)
        nrmse = np.sqrt(np.sum(absolute_error ** 2)) / np.sqrt(np.sum(real ** 2))
    metrics = {
        'kendall_tau': float(kendalltau(real, predicted).statistic),
        'MAPE': float(np.mean(relative_error)),
        'Acc_10': float(np.mean(relative_error <= 0.10)),
        'NMAE': float(nmae),
        'NRMSE': float(nrmse),
        'spearman': float(spearmanr(real, predicted).statistic),
    }
    return metrics
