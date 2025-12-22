# coding : utf-8
# Author : yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年3月9日17:47:08）

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, config):
        self.config = config
        self.monitor_reverse = config.monitor_reverse
        self.patience = config.patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 1e9
        self.delta = 0
        self.best_model = None
        self.best_epoch = None

    def __call__(self, epoch, params, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, params, val_loss)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_epoch = epoch
            self.best_score = score
            self.save_checkpoint(epoch, params, val_loss)
            self.counter = 0

    def track(self, epoch, params, error):
        self.__call__(epoch, params, error)

    def save_checkpoint(self, epoch, params, val_loss):
        # 你这里原来是 epoch + 1，我保留不动（配合你外部的 sum_time 逻辑）
        self.best_epoch = epoch + 1

        # ✅ 关键修复：深拷贝 state_dict，避免后续训练把 best_model “污染”
        # 同时搬到 CPU，节省显存，load_state_dict 也完全兼容
        self.best_model = {k: v.detach().cpu().clone() for k, v in params.items()}

        self.val_loss_min = val_loss


    def track_one_epoch(self, epoch, model, error, metric):
        # ✅ 兼容 DataParallel：取真实模型的 state_dict
        real_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        state = real_model.state_dict()

        if self.monitor_reverse:
            self.track(epoch, state, -1 * error[metric])
        else:
            self.track(epoch, state, error[metric])
