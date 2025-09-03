# coding : utf-8
# Author : Yuxiang Zeng
import torch
import torch.nn as nn
import random


# 在这里加上每个Batch的loss，如果有其他的loss，请在这里添加，
def compute_loss(model, pred, label, config):
    pred = pred.reshape(label.shape)
    
    if config.model == 'ours':
        return model.loss_function(pred, label) + 0.5 * model.rank_loss(pred, label) + 0.1 * model.ac_loss(pred)
    
    loss = model.loss_function(pred, label)
    return loss
    
    # if config.model == 'nnformer':
        # loss = model.nnformer_loss(pred, label)
        # return loss