# coding : utf-8
# Author : yuxiang Zeng

import os
import platform
import time
import logging
import pickle
import numpy as np
import torch
from utils.model_efficiency import *


class Logger:
    def __init__(self, filename, exper_detail, plotter, config, show_params=True):
        self.filename = filename
        self.exper_detail = exper_detail
        self.plotter = plotter
        self.config = config
        self._init_log_file()
        if config.hyper_search:
            self.exper_filename += '_hyper_search'
        # 设置日志记录到文件
        logging.basicConfig(level=logging.INFO, filename=f"{self.exper_filename}.md", filemode='w', format='%(message)s')
        self.logger = logging.getLogger(config.model)
        config.log = self
        self._prepare_experiment(show_params)

    # 初始化日志文件路径
    def _init_log_file(self):
        fileroot = f'./results/{self.config.model}/' + time.strftime('%Y%m%d') + '/log/'
        os.makedirs(fileroot, exist_ok=True)
        timestamp = time.strftime('%H_%M_%S')
        self.exper_filename = os.path.join(fileroot, f"{timestamp}_{self.filename}")

    # 打印初始配置参数
    def _prepare_experiment(self, show_params):
        self.logger.info('```python')
        if show_params:
            self.log(self._format_config_dict(self.config.__dict__))

    # 保存运行日志到run.log文件
    def save_in_log(self, metrics):
        # === 获取 CPU 名称 ===
        cpu_name = platform.processor()
        if not cpu_name:
            cpu_name = platform.machine()
        device_name = f"CPU-{cpu_name}"

        # === 如果有 GPU，添加 GPU 名称 ===
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            device_name += f"_GPU-{gpu_name}"

        # === 清理非法字符，生成文件名 tag ===
        device_tag = device_name.replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "")

        # === 构造日志文件路径 ===
        log_path = f"./{device_tag}_{self.config.logger}"

        with open(f"./{log_path}.log", 'a') as f:
            timestamp = time.strftime('|%Y-%m-%d %H:%M:%S| ')
            f.write(timestamp + self.exper_detail + '\n')
            metric_str = ' '.join([f"{k} - {np.mean(v):.4f}" for k, v in metrics.items()])
            f.write(timestamp + metric_str + '\n')

    # 保存结果到pickle文件
    def save_result(self, metrics):
        os.makedirs('./results/metrics/', exist_ok=True)
        config_copy = {k: v for k, v in self.config.__dict__.items() if k != 'log'}
        result = {
            'config': config_copy,
            'dataset': self.config.model,
            'model': self.config.model,
            **{k: metrics[k] for k in metrics},
            **{f"{k}_mean": np.mean(metrics[k]) for k in metrics},
            **{f"{k}_std": np.std(metrics[k]) for k in metrics},
        }
        with open(f'./results/metrics/{self.filename}.pkl', 'wb') as f:
            pickle.dump(result, f)

    # 日志输出（含彩色打印）
    def log(self, string):
        if string.startswith('\n'):
            string = string[1:]
            print('\n', end='')
            self.logger.info('')
        timestamp = time.strftime('|%Y-%m-%d %H:%M:%S| ')
        self.logger.info(timestamp + string)
        self.only_print(string)

    def __call__(self, string):
        self.log(string)

    # 终端彩色输出辅助函数
    def only_print(self, string):
        timestamp = time.strftime('|%Y-%m-%d %H:%M:%S| ')
        print(f"\033[1;38;2;151;200;129m{timestamp}\033[0m\033[1m{string}\033[0m")

    # 展示一次完整实验结果
    def show_results(self, result_error, sum_time):
        monitor = self.config.monitor_metrics
        summary = f"Valid{monitor}={-result_error[monitor]:.4f} ｜ "
        summary += ' '.join([f"{k}={v:.4f}" for k, v in result_error.items()])
        summary += f" time={sum_time:.1f} s"
        self.only_print(summary)

    # 展示训练中的某轮 epoch 的误差
    def show_epoch_error(self, runId, epoch, monitor, epoch_loss, result_error, train_time):
        if self.config.verbose and epoch % self.config.verbose == 0:
            self.only_print(self.exper_detail)
            best = f"Best Epoch {monitor.best_epoch} {self.config.monitor_metric} = {-monitor.best_score:.4f}  now = {epoch - monitor.best_epoch}"
            self.only_print(best)
            summary = f"Round={runId + 1} Epoch={epoch + 1:03d} Loss={epoch_loss:.4f} "
            summary += ' '.join([f"v{k}={v:.4f}" for k, v in result_error.items()])
            summary += f" time={sum(train_time):.1f} s"
            self.only_print(summary)

    # 展示最终测试结果
    def show_test_error(self, runId, monitor, result_error, sum_time):
        summary = f"Round={runId + 1} BestEpoch={monitor.best_epoch:3d} "
        summary += f"Valid{self.config.monitor_metric}={-monitor.best_score:.4f} ｜ "
        summary += ' '.join([f"{k}={v:.4f}" for k, v in result_error.items()])
        summary += f" time={sum_time:.1f} s"
        self.log(summary)

    # 配置参数格式化输出
    def _format_config_dict(self, config_dict, items_per_line=3):
        sorted_items = sorted(config_dict.items())
        lines = [', '.join([f"'{k}': {v}" for k, v in sorted_items[i:i + items_per_line]]) for i in range(0, len(sorted_items), items_per_line)]
        return '{\n' + '\n'.join(['     ' + line for line in lines]) + '\n}'

    
    
    # 实验结束时执行的清理操作
    def end_the_experiment(self, model):
        self.logger.info(f'\n{str(model)}')
        self.logger.info('```')
        self._delete_empty_directories('./results/')
        
        
    


