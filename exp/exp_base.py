# coding : utf-8
# Author : Yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年3月27日23:33:32）
import contextlib
import torch
from time import time

from exp.exp_loss import compute_loss
from exp.exp_metrics import ErrorMetrics
from utils.model_trainer import get_loss_function, get_optimizer
from torch.cuda.amp import autocast
from tqdm import *


class BasicModel(torch.nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.config = config
        self.scaler = torch.amp.GradScaler(config.device)  # ✅ 初始化 GradScaler
        
    def forward(self, *x):
        y = self.model(*x)
        return y

    def setup_optimizer(self, config):
        self.to(config.device)
        self.loss_function = get_loss_function(config).to(config.device)
        self.optimizer     = get_optimizer(self.parameters(), lr=config.lr, decay=config.decay, config=config)
        self.scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=config.patience // 1.5, threshold=0.0
        )

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time()

        for train_batch in (dataModule.train_loader):
            all_item = [item.to(self.config.device) for item in train_batch]
            inputs, label = all_item[:-1], all_item[-1]

            self.optimizer.zero_grad()

            if self.config.use_amp:
                with torch.amp.autocast(device_type=self.config.device):
                    pred = self.forward(*inputs)
                    loss = compute_loss(self, pred, label, self.config)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.forward(*inputs)
                loss = compute_loss(self, pred, label, self.config)
                loss.backward()
                self.optimizer.step()

            # for i in range(len(pred)):
                # print(f'Pred: {pred[i].item():.4f}, Real: {label[i].item():.4f}')
        t2 = time()
        return loss, t2 - t1

    def evaluate_one_epoch(self, dataModule, mode='valid'):
        self.eval()
        torch.set_grad_enabled(False)
        dataloader = dataModule.valid_loader if mode == 'valid' and len(dataModule.valid_loader.dataset) != 0 else dataModule.test_loader
        preds, reals, val_loss = [], [], 0.

        context = (
            torch.amp.autocast(device_type=self.config.device)
            if self.config.use_amp else
            contextlib.nullcontext()
        )
        with context:
            for batch in dataloader:
                all_item = [item.to(self.config.device) for item in batch]
                inputs, label = all_item[:-1], all_item[-1]
                pred = self.forward(*inputs)

                if mode == 'valid':
                    val_loss += compute_loss(self, pred, label, self.config)

                if self.config.classification:
                    pred = torch.max(pred, 1)[1]
                    
                preds.append(pred)
                reals.append(label)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        
        if self.config.dataset != 'weather':
            reals, preds = dataModule.y_scaler.inverse_transform(reals), dataModule.y_scaler.inverse_transform(preds)
            
        # for i in range(len(preds)):
            # print(f'Pred: {preds[i].item():.4f}, Real: {reals[i].item():.4f}')
            
        if mode == 'valid':
            self.scheduler.step(val_loss)

        return ErrorMetrics(reals, preds, self.config)