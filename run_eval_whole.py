# coding : utf-8
# Author : Yuxiang Zeng
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import torch
import collections
from data_provider.data_center import DataModule
from exp.exp_model import Model
import exp.exp_efficiency
import utils.utils
import pickle

torch.set_default_dtype(torch.float32)


def get_experiment_name(config):
    # === 构建 exper_detail 字典（基础字段）===
    detail_fields = {
        "Model": config.model,
        "dataset": config.dataset,
        "spliter_ratio": config.spliter_ratio,
        "d_model": config.d_model,
    }

    # === 动态添加字段（只有在 config 中存在才加入）===
    optional_fields = ["idx", "op_encoder"]
    for field in optional_fields:
        if hasattr(config, field):
            key = (
                field.replace("_", " ").title().replace(" ", "_")
            )  # e.g. seq_len -> Seq_Len
            detail_fields[key] = getattr(config, field)

    # === 构建字符串 ===
    exper_detail = ", ".join(f"{k} : {v}" for k, v in detail_fields.items())
    log_filename = "_".join(
        f"{k.replace('_', '')}{v}" for k, v in detail_fields.items()
    )

    return log_filename, exper_detail


# coding : utf-8
# Author : Yuxiang Zeng
import os
import torch
from tqdm import *
import pickle
from exp.exp_base import BasicModel, exec_evaluate_one_epoch, exec_evaluate_whole_dataset, exec_train_one_epoch
from utils.model_monitor import EarlyStopping
from utils.exp_logger import Logger
from data_provider.data_center import DataModule


def RunOnce(config, runid, model: BasicModel, datamodule: DataModule, log: Logger):
    torch.set_float32_matmul_precision("high")

    # 创建保存模型的目录
    os.makedirs(f"./checkpoints/{config.model}", exist_ok=True)
    model_path = f"./checkpoints/{config.model}/{log.log_filename}_round_{runid}.pt"

    # 判断是否需要重新训练：
    # 若 config.retrain==1 表示强制重训；
    # 或者模型文件不存在 且 设置了 continue_train，则需要重新训练
    retrain_required = (
        config.retrain == 1 or not os.path.exists(model_path) and config.continue_train
    )

    # 如果无需重新训练且已有模型文件，则直接加载模型并评估测试集性能
    try:
        # 加载之前记录的训练时间
        sum_time = pickle.load(
            open(f"./results/metrics/" + log.log_filename + ".pkl", "rb")
        )["train_time"][runid]
        # 加载模型权重（weights_only=True 可忽略 optimizer 等无关信息）
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location="cpu")
        )
        model.setup_optimizer(config)  # 重新设置优化器
        results = exec_evaluate_whole_dataset(model, datamodule)  # 在测试集评估性能
        log.show_results(results, sum_time)
        config.record = False  # 不再记录当前结果
    except Exception as e:
        log.only_print(f"Error: {str(e)}")
        retrain_required = True  # 若加载失败则触发重新训练
        raise e

    return results


def RunExperiments(log, config):
    log("*" * 20 + "Experiment Start" + "*" * 20)
    metrics = collections.defaultdict(list)

    for runid in range(config.rounds):
        config.runid = runid
        log.set_runid(runid)
        utils.utils.set_seed(config.seed + runid)

        datamodule = DataModule(config)
        model = Model(config)

        log.plotter.reset_round()
        results = RunOnce(config, runid, model, datamodule, log)
        log.log_model_graph(
            model, datamodule, config.device
        )  # 记录模型图 (可能会报错，如果不兼容可以直接注释)

        for key in results:
            metrics[key].append(results[key])

    log("*" * 20 + "Experiment Results:" + "*" * 20)
    log(log.exper_detail)
    log(
        f"Train_length : {len(datamodule.train_loader.dataset)} Valid_length : {len(datamodule.valid_loader.dataset)} Test_length : {len(datamodule.test_loader.dataset)}"
    )

    for key in metrics:
        log(f"{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}")

    log.save_in_log(metrics)
    log.save_result(metrics)
    log.plotter.record_metric(metrics)

    log("*" * 20 + "Experiment Success" + "*" * 20)
    log.end_the_experiment(model)
    return metrics


if __name__ == "__main__":
    # Experiment Settings, logger, plotter
    from utils.exp_logger import Logger
    from utils.exp_metrics_plotter import MetricsPlotter
    from utils.utils import set_settings
    from utils.exp_config import get_config

    config = get_config("OurModelConfig")
    set_settings(config)

    if config.dataset == "nnlqp":
        config.predict_target = "latency"
        config.sample_method = "nnlqp"
        config.input_size = 32
        config.bs = 32
        config.tqdm = True
        config.epochs = 200
        config.monitor_metric = "MAPE"
        config.monitor_reverse = False

        if config.model == "narformer":
            config.input_size = 1216
            config.graph_d_model = 960
            config.d_model = 1216
    elif config.dataset == "101_acc":
        config.bs = 1024
        config.sample_method = "random"
        # config.spliter_ratio = '1:4:95'  #  '0.02:4:95.98' '0.04:4:95.96' '0.1:4:95.9'

    if config.try_exp in [3, 4]:
        config.scale = False

    log_filename, exper_detail = get_experiment_name(config)
    plotter = MetricsPlotter(log_filename, config)
    log = Logger(log_filename, exper_detail, plotter, config)
    metrics = RunExperiments(log, config)
