# coding : utf-8
# Author : Yuxiang Zeng
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import torch
import collections
from data_provider.data_center import DataModule
from exp.exp_train import RunOnce
from exp.exp_model import Model
import exp.exp_efficiency
import utils.utils
torch.set_default_dtype(torch.float32)

def get_experiment_name(config):
    # === 构建 exper_detail 字典（基础字段）===
    detail_fields = {
        'Model': config.model,
        'dataset': config.dataset,
        # 'dst_dataset': config.dst_dataset.split('/')[-1].split('.')[0],
        'spliter_ratio': config.spliter_ratio,
        'd_model': config.d_model,
    }

    # === 动态添加字段（只有在 config 中存在才加入）===
    optional_fields = ['idx', 'op_encoder']
    for field in optional_fields:
        if hasattr(config, field):
            key = field.replace('_', ' ').title().replace(' ', '_')  # e.g. seq_len -> Seq_Len
            detail_fields[key] = getattr(config, field)

    # === 构建字符串 ===
    exper_detail = ', '.join(f"{k} : {v}" for k, v in detail_fields.items())
    log_filename = '_'.join(f"{k.replace('_', '')}{v}" for k, v in detail_fields.items())

    return log_filename, exper_detail


def RunExperiments(log, config):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runid in range(config.rounds):
        config.runid = runid
        utils.utils.set_seed(config.seed + runid)
        datamodule = DataModule(config)
        model = Model(config)
        if runid == 0:
            exp.exp_efficiency.evaluate_model_efficiency(datamodule, model, log, config)
        log.plotter.reset_round()
        results = RunOnce(config, runid, model, datamodule, log)
        for key in results:
            metrics[key].append(results[key])
        log.plotter.append_round()

    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    log(log.exper_detail)
    log(f'Train_length : {len(datamodule.train_loader.dataset)} Valid_length : {len(datamodule.valid_loader.dataset)} Test_length : {len(datamodule.test_loader.dataset)}')

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')
        
    log.save_in_log(metrics)

    if config.record:
        log.save_result(metrics)
        log.plotter.record_metric(metrics)
    log('*' * 20 + 'Experiment Success' + '*' * 20)
    log.end_the_experiment(model)
    return metrics


if __name__ == '__main__':
    # Experiment Settings, logger, plotter
    from utils.exp_logger import Logger
    from utils.exp_metrics_plotter import MetricsPlotter
    from utils.utils import set_settings
    from utils.exp_config import get_config
    # config = get_config('FlopsConfig')
    # config = get_config('MacConfig')
    # config = get_config('LSTMConfig')
    # config = get_config('GRUConfig')
    # config = get_config('BRPNASConfig')
    # config = get_config('GATConfig')
    # config = get_config('NarFormerConfig')
    # config = get_config('NarFormer2Config')
    # config = get_config('NNformerConfig')
    config = get_config('OurModelConfig')
    set_settings(config)
    
    if config.dataset == 'nnlqp':
        config.input_size = 29
        if config.model == 'narformer':
            config.input_size = 1216
            config.graph_d_model = 960
            config.d_model = 1216
    elif config.dataset == '101_acc':
        config.bs = 1024
            
    log_filename, exper_detail = get_experiment_name(config)
    plotter = MetricsPlotter(log_filename, config)
    log = Logger(log_filename, exper_detail, plotter, config)
    metrics = RunExperiments(log, config)
