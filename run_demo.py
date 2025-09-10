# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
from utils.exp_sh import once_experiment
from datetime import datetime

# 在这里写下超参数探索空间
hyper_dict = {
    'op_encoder': ['embedding'],  # 'embedding', 'nerf', 'nape'
    'num_layers': [3],
    'd_model': [256, 128, 192,],
}

######################################################################################################
# 这里是总执行实验顺序！！！！！！！！
def experiment_run():
    Our_model()
    return True


def Ablation():
    hyper_dict = {
        # 'dataset': ['a', 'b', 'c'],
        'ablation': [1, 2, 3],
    }
    once_experiment('TestConfig', hyper_dict, grid_search=1, retrain=1)
    return True


def Our_model(hyper=None):
    # monitor_metric = NMAE KendallTau
    once_experiment('OurModelConfig', hyper_dict, monitor_metric='KendallTau', reverse=True)
    return True


def log_message(message):
    log_file = "run.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

if __name__ == "__main__":
    try:
        log_message("Experiment Start!!!")
        experiment_run()
    except KeyboardInterrupt as e:
        log_message("Experiment interrupted by user.")
    finally:
        log_message("All commands executed successfully.\n")


