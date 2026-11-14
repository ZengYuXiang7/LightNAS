"""单轮延迟预测实验：数据、训练、验证和测试。"""

import sys
import time

import torch
from tqdm import tqdm

from data_provider.data import DataModule, experiment
from models.select_model import Model
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.observability import result_checkpoint_path
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import set_seed


class ExpLatency:
    def __init__(self, args, run_id, result_name, log):
        self.args = args
        self.run_id = run_id
        self.log = log

        # 每轮沿用原种子、数据划分和模型初始化顺序。
        set_seed(args.seed + run_id)
        data_experiment = experiment(args)
        self.data = DataModule(data_experiment, args)
        self.model = Model(self.data, args)
        self.monitor = EarlyStopping(args)
        self.model_path = result_checkpoint_path(args.dataset_name, result_name, run_id + 1)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # 验证损失的调度方向与原回归实现保持一致。
        self.model.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.model.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=args.patience // 1.5, threshold=0.0)

    def train_epoch(self):
        # 每个批次按原顺序更新参数，并返回最后一批损失及本轮耗时。
        self.model.train()
        torch.set_grad_enabled(True)
        start_time = time.time()
        for batch in self.data.train_loader:
            graph, features, value = batch
            graph, features, value = graph.to(self.args.device), features.to(self.args.device), value.to(self.args.device)
            prediction = self.model.forward(graph, features)
            loss = self.loss_function(prediction, value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elapsed_time = time.time() - start_time
        self.model.eval()
        torch.set_grad_enabled(False)
        return loss, elapsed_time

    def collect_predictions(self, dataloader, validation):
        # 验证时累计损失；验证和测试共同收集预测及真实标签。
        val_loss = 0.0
        predictions = []
        targets = []
        for batch in dataloader:
            graph, features, value = batch
            graph, features, value = graph.to(self.args.device), features.to(self.args.device), value.to(self.args.device)
            prediction = self.model.forward(graph, features)
            if validation:
                val_loss += self.loss_function(prediction, value)
            predictions.append(prediction)
            targets.append(value)
        targets = torch.cat(targets, dim=0)
        predictions = torch.cat(predictions, dim=0)
        return targets, predictions, val_loss

    def validate(self):
        # 验证损失驱动学习率调度，指标还原到原始延迟尺度。
        targets, predictions, val_loss = self.collect_predictions(self.data.valid_loader, True)
        self.scheduler.step(val_loss)
        real = targets * self.data.max_value
        predicted = predictions * self.data.max_value
        metrics = ErrorMetrics(real, predicted)
        return metrics, metrics['NMAE']

    def test(self):
        # 使用本轮训练得到的最优权重计算测试指标。
        targets, predictions, _ = self.collect_predictions(self.data.test_loader, False)
        metrics = ErrorMetrics(targets * self.data.max_value, predictions * self.data.max_value)
        return metrics

    def train(self):
        # 按轮次执行训练、验证、早停和最优权重恢复。
        train_time = []
        show_progress = self.args.tqdm and sys.stderr.isatty()
        progress = tqdm(total=self.args.epochs, desc=f'Round {self.run_id + 1}', disable=not show_progress)
        for epoch in range(self.args.epochs):
            epoch_start = time.perf_counter()
            epoch_loss, time_cost = self.train_epoch()
            valid_error, valid_nmae = self.validate()
            self.monitor.track_one_epoch(epoch, self.model, valid_nmae)
            train_time.append(time_cost)
            if show_progress:
                progress.set_postfix(loss=f'{epoch_loss:.4f}', valid_nmae=f'{valid_nmae:.4f}')
                progress.update()
            completed = epoch + 1
            if completed == 1 or completed % self.args.print_freq == 0:
                self.log.show_epoch_error(self.run_id, epoch, self.monitor, epoch_loss, valid_error, train_time)
            elif completed == self.args.epochs or self.monitor.early_stop:
                self.log.show_epoch_error(self.run_id, epoch, self.monitor, epoch_loss, valid_error, train_time)
            if completed == 1 and time.perf_counter() - epoch_start <= 5:
                progress.close()
                show_progress = False
            if self.monitor.early_stop:
                self.log(f'Round {self.run_id + 1}: early stopping at epoch {completed}.')
                break
        progress.close()
        self.model.load_state_dict(self.monitor.best_model)
        sum_time = sum(train_time[:self.monitor.best_epoch])
        return sum_time
