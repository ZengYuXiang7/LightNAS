# coding : utf-8
# Author : Yuxiang Zeng
import torch
import torch.nn as nn
import random


# 在这里加上每个Batch的loss，如果有其他的loss，请在这里添加，
def compute_loss(model, pred, label, config):
    pred = pred.reshape(label.shape)
    
    loss = model.loss_function(pred, label)
    
    # 如果 config.rank_loss 为 True，则追加排序损失
    # if getattr(config, 'rank_loss', False):
    #     # print(f"Rank loss {model.rank_loss(pred, label)} enabled, adding to total loss.")
    #     loss += model.rank_loss(pred, label) * 0.1
    
    # if getattr(config, 'ac_loss', False):
    #     # 如果存在 ac_loss，则计算并添加到总损失中
    #     # print(f"AC loss {model.ac_loss(pred)} enabled, adding to total loss.")
    #     loss += model.ac_loss(pred) * 0.5
        
    return loss
