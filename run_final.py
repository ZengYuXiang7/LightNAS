# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
from utils.exp_sh import once_experiment
from datetime import datetime
import argparse

# 在这里写下超参数探索空间
hyper_dict = {
    "d_model": [128, 192, 256],
    "num_layers": [4, 5, 6],
    "scale": [0],
}


######################################################################################################
# 这里是总执行实验顺序！！！！！！！！
def experiment_run(args):
    # bench201 ["1:4:95", "3:4:93", "5:4:91", "10:4:86"]
    # bench101 ["0.025:4:95.9775", "0.04:4:95.96", "0.1:4:95.9", "1:4:95"]
    # for data_split in ["0.025:4:95.9775", "0.04:4:95.96", "0.1:4:95.9", "1:4:95"]:
    for data_split in ["1:4:95", "3:4:93", "5:4:91", "10:4:86"]:
        now_hyper_dict = {
            "spliter_ratio": [data_split],
            **hyper_dict,
        }
        Our_model(now_hyper_dict, args)
    return True


def Our_model(hyper, args):
    # monitor_metric = NMAE KendallTau
    once_experiment(
        "OurModelConfig",
        hyper,
        monitor_metric="KendallTau",
        reverse=True,
        debug=0,
        grid_search=0,
        args=args,
    )
    return True


def log_message(message):
    log_file = "run.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=bool, default=False)
    args = parser.parse_args()
    try:
        log_message("Experiment Start!!!")
        experiment_run(args)
    except KeyboardInterrupt as e:
        log_message("Experiment interrupted by user.")
    finally:
        log_message("All commands executed successfully.\n")
