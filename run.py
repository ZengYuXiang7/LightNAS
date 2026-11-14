# coding : utf-8
# Author : Anonymous
import collections
import sys
from pathlib import Path

import numpy as np
import torch

from exp.exp_latency import ExpLatency
from utils.ml import run_ml_experiments
from utils.result_contract import build_data_config, publish_result

sys.dont_write_bytecode = True
torch.set_default_dtype(torch.float32)


def RunOnce(args, run_id, log, log_filename):
    # 初始化本轮数据和模型，完整执行训练、测试与保存。
    exp = ExpLatency(args, run_id, log_filename, log)
    sum_time = exp.train()
    results = exp.test()
    log.show_test_error(run_id, exp.monitor, results, sum_time)
    torch.save(exp.monitor.best_model, exp.model_path)
    log.only_print(f'Model parameters saved to {exp.model_path}')
    best_epoch = exp.monitor.best_epoch
    best_valid = {'NMAE': float(-exp.monitor.best_score)}
    early_stop = {'patience': args.patience, 'stopped': exp.monitor.early_stop, 'counter': exp.monitor.counter}

    # 每轮记录对应种子、验证指标、测试指标和实际保存的权重。| 指标 | 含义 | 方向 |
    # |---|---|---|
    # | Kendall Tau（KT） | 衡量预测延迟与真实延迟的成对排序是否一致 | 越大越好 |
    # | Spearman | 衡量预测延迟与真实延迟的排名相关性 | 越大越好 |
    # | MAPE | 平均绝对相对误差，衡量预测偏离真实延迟的比例 | 越小越好 |
    # | Acc_10 | 相对误差不超过 10% 的样本占比 | 越大越好 |
    # | MAE | 平均绝对误差，单位与延迟标签一致，通常为毫秒 | 越小越好 |
    test_metrics = {}
    for name, value in results.items():
        test_metrics[name] = float(value)
    round_record = {
        'seed': args.seed + run_id,
        'best_epoch': best_epoch,
        'best_valid': best_valid,
        'test_metrics': test_metrics,
        'checkpoint': exp.model_path.as_posix(),
        'train_time': float(sum_time),
        'early_stop': early_stop,
    }
    data_config = build_data_config(args, len(exp.data.y), len(exp.data.train_y), len(exp.data.valid_y), len(exp.data.test_y))
    return round_record, data_config


def RunExperiments(log, args, log_filename):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)
    rounds = []

    for run_id in range(args.rounds):
        round_record, data_config = RunOnce(args, run_id, log, log_filename)
        rounds.append(round_record)
        for key, value in round_record['test_metrics'].items():
            metrics[key].append(value)
        metrics['train_time'].append(round_record['train_time'])
        if args.record:
            publish_result(args, log_filename, args.model, data_config, rounds, False)

    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')
    if args.record:
        publish_result(args, log_filename, args.model, data_config, rounds, True)
    log('*' * 20 + 'Experiment Success' + '*' * 20)
    return metrics


def main(args, log, log_filename):
    # 机器学习使用独立回归器与验证集选择，其余模型沿用神经网络训练。
    if args.model == 'ml':
        metrics = run_ml_experiments(args, log, log_filename)
    else:
        metrics = RunExperiments(log, args, log_filename)
    # log.send_email(log_filename, metrics, '22rqwang@stu.edu.cn')
    log.end_the_experiment()


if __name__ == '__main__':
    # Experiment settings and logger
    from config import get_config
    from utils.logger import Logger
    from utils.utils import set_settings
    args = get_config()
    set_settings(args)
    # log_filename = f'Model_{args.model}_{args.dataset}_S{args.train_size}_R{args.rank}_Ablation{args.Ablation}'
    if args.result_name:
        log_filename = args.result_name
    elif args.model == 'ml':
        batch_name = '-'.join(Path(args.batch_dir).parts)
        log_filename = f'{batch_name}-Model_ml_gradient_boosting_{args.dataset}_{args.train_device}_S{args.train_size}_seed{args.seed}_n{args.rounds}'
    else:
        batch_name = '-'.join(Path(args.batch_dir).parts)
        log_filename = f'{batch_name}-Model_{args.model}_{args.dataset}_{args.train_device}_S{args.train_size}_R{args.rank}_seed{args.seed}_n{args.rounds}'
    log = Logger(args)
    main(args, log, log_filename)
