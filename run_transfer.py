
# coding : utf-8
# Author : Yuxiang Zeng
import numpy as np
import torch
import collections
from data_provider.data_center import DataModule
from exp.exp_train import RunOnce
from exp.exp_model import Model
from run_train import get_experiment_name
import exp.exp_efficiency
import utils.utils
import os
import torch
from tqdm import *
import pickle
from utils.model_monitor import EarlyStopping
torch.set_default_dtype(torch.float32)

def get_pretrained_model(model, runid, config):
    config.dst_dataset, config.src_dataset = config.src_dataset, config.src_dataset
    log_filename, _ = get_experiment_name(config)
    model_path = f'./checkpoints/{config.model}/{log_filename}_round_{runid}.pt'
    
    print(model_path)
    # 这里的缩进使用4个空格
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    config.dst_dataset, config.src_dataset = config.src_dataset, config.src_dataset

    return model


def run_transfer(config, runid, model, datamodule, log):
    try:
        # 一些模型（如Keras兼容模型）可能需要compile，跳过非必要的compile
        model.compile()
    except Exception as e:
        print(f'Skip the model.compile() because {e}')

    # 设置EarlyStopping监控器
    monitor = EarlyStopping(config)

    # 创建保存模型的目录
    os.makedirs(f'./checkpoints/{config.model}', exist_ok=True)
    model_path = f'./checkpoints/{config.model}/{log.filename}_round_{runid}.pt'

    model = get_pretrained_model(model, runid, config)
    model.setup_optimizer(config)
    
    train_time = []
    for epoch in trange(config.epochs):
        if monitor.early_stop:
            break  # 若满足early stopping条件则提前终止训练

        # 训练一个epoch并记录耗时
        train_loss, time_cost = model.train_one_epoch(datamodule)
        train_time.append(time_cost)

        # 验证集上评估当前模型误差
        valid_error = model.evaluate_one_epoch(datamodule, 'valid')

        # 将当前epoch的验证误差传递给early stopping模块进行跟踪
        monitor.track_one_epoch(epoch, model, valid_error, config.monitor_metric)

        # 输出当前epoch的训练误差和验证误差，并记录训练时间
        log.show_epoch_error(runid, epoch, monitor, train_loss, valid_error, train_time)

        # 更新日志可视化（如绘图）
        log.plotter.append_epochs(train_loss, valid_error)

        # 暂存模型参数（即使不是最优，也为了中断续训做准备）
        torch.save(model.state_dict(), model_path)


    # 加载最优模型参数（来自early stopping）
    model.load_state_dict(monitor.best_model)

    # 累计训练时间（仅使用前best_epoch轮）
    sum_time = sum(train_time[: monitor.best_epoch])

    # 使用最优模型在测试集评估
    results = model.evaluate_one_epoch(datamodule, 'test')
    # results = {f'Valid{config.monitor_metric}': abs(monitor.best_score), **results}
    log.show_test_error(runid, monitor, results, sum_time)

    # 保存最优模型参数
    torch.save(monitor.best_model, model_path)
    log(f'Model parameters saved to {model_path}')

    # 将训练时间加入返回结果中
    results['train_time'] = sum_time
    return results


def RunExperiments(log, config):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runid in range(config.rounds):
        config.runid = runid
        utils.utils.set_seed(config.seed + runid)
        datamodule = DataModule(config)
        model = Model(config)
        log.plotter.reset_round()
        results = run_transfer(config, runid, model, datamodule, log)
        for key in results:
            metrics[key].append(results[key])
        log.plotter.append_round()

    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    log(log.exper_detail)
    log(f'Train_length : {len(datamodule.train_loader.dataset)} Valid_length : {len(datamodule.valid_loader.dataset)} Test_length : {len(datamodule.test_loader.dataset)}')

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')
    try:
        flops, params, inference_time = utils.model_efficiency.get_efficiency(datamodule, Model(config), config)
        log(f"Flops: {flops:.0f}")
        log(f"Params: {params:.0f}")
        log(f"Inference time: {inference_time:.2f} ms")
    except Exception as e:
        log('Skip the efficiency calculation')

    log.save_in_log(metrics)

    if config.record:
        log.save_result(metrics)
        log.plotter.record_metric(metrics)
    log('*' * 20 + 'Experiment Success' + '*' * 20)
    log.end_the_experiment(model)
    return metrics


def run(config):
    from utils.exp_logger import Logger
    from utils.exp_metrics_plotter import MetricsPlotter
    from utils.utils import set_settings
    set_settings(config)
    log_filename, exper_detail = get_experiment_name(config)
    plotter = MetricsPlotter(log_filename, config)
    log = Logger(log_filename, exper_detail, plotter, config)
    metrics = RunExperiments(log, config)
    return metrics


if __name__ == '__main__':
    # Experiment Settings, logger, plotter
    from utils.exp_config import get_config
    # config = get_config('GNNModelConfig')
    # config = get_config('TransModelConfig')
    config = get_config('NarFormerConfig')

    if config.dataset == 'nnlqp':
        config.input_size = 29
        if config.model == 'narformer':
            config.input_size = 1216
            config.graph_d_model = 960
            config.d_model = 1216
    elif config.dataset == 'nasbench201':
        if config.model not in ['ours', 'narformer']:
            config.input_size = 6

    run(config)
