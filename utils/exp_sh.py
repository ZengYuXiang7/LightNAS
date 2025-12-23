# coding : utf-8
# Author : yuxiang Zeng
import time
import subprocess
import numpy as np
from datetime import datetime
import pickle
from itertools import product

from run_train import get_experiment_name
from utils.exp_config import get_config


def write_and_print(string):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("./run.log", "a") as f:
        print(string)
        f.write(f"[{timestamp}] {string}\n")
        # f.write(string + '\n')

    return True


# 搜索最佳超参数然后取最佳
def add_parameter(command: str, params: dict) -> str:
    for param_name, param_value in params.items():
        command += f" --{param_name} {param_value}"
    return command


def once_experiment(
    exper_name,
    hyper_dict,
    monitor_metric,
    reverse=False,
    grid_search=0,
    retrain=1,
    debug=0,
    run_again=False,
    args=None
):
    # 先进行超参数探索
    best_hyper = hyper_search(
        exper_name,
        hyper_dict,
        monitor_metric=monitor_metric,
        reverse=reverse,
        grid_search=grid_search,
        retrain=retrain,
        debug=debug,
        args=args
    )

    if run_again:
        # 再跑最佳参数实验
        commands = []
        command = f"python run_train.py --exp_name {exper_name} --retrain 1"
        commands.append(command)

        commands = [add_parameter(command, best_hyper) for command in commands]

        # 执行所有命令
        for command in commands:
            run_command(command)
    return True


def hyper_search(
    exp_name,
    hyper_dict,
    monitor_metric,
    reverse=False,
    grid_search=0,
    retrain=1,
    debug=0,
    args=None
):
    """
    入口函数：选择使用网格搜索还是逐步搜索
    """
    if grid_search:
        return grid_search_hyperparameters(
            exp_name, hyper_dict, retrain, monitor_metric, reverse, debug, args
        )
    else:
        return sequential_hyper_search(
            exp_name, hyper_dict, retrain, monitor_metric, reverse, debug, args
        )


def run_and_get_metric(cmd_str, config, chosen_hyper, monitor_metric, debug=False, args=None):
    """
    运行训练命令，并提取 metric
    """
    timestamp = time.strftime("|%Y-%m-%d %H:%M:%S| ")
    print(
        f"\033[1;38;2;151;200;129m{timestamp}\033[0m \033[1;38;2;100;149;237m{cmd_str}\033[0m"
    )
    config.__dict__.update(chosen_hyper)
    log_filename = get_experiment_name(config)[0]

    print(log_filename, chosen_hyper)
    if args.experiment:
        cmd_str += f"--experiment 1 "
    subprocess.run(cmd_str, shell=True)

    metric_file_address = f"./results/metrics/" + get_experiment_name(config)[0]
    this_expr_metrics = pickle.load(open(metric_file_address + ".pkl", "rb"))

    # 选择最优 metric
    best_value = np.mean(this_expr_metrics[monitor_metric])
    return best_value


def sequential_hyper_search(
    exp_name, hyper_dict, retrain, monitor_metric, reverse, debug, args
):
    """
    逐步搜索超参数，每次调整一个参数，并保持其他最优值
    - 修复：避免后续超参的第一个值重复执行
    - 修复：不再把未探索超参写进 best_hyper（避免副作用）
    - 新增：evaluated_cache，相同配置只跑一次
    """
    config = get_config(exp_name)
    log_file = f"./run.log"
    best_hyper = {}

    # 缓存：同一组超参组合 -> metric
    evaluated_cache = {}

    def make_key(d: dict):
        return tuple(sorted(d.items()))

    def run_once_with_cache(chosen_dict: dict):
        key = make_key(chosen_dict)
        if key in evaluated_cache:
            return evaluated_cache[key]

        command = f"python run_train.py --exp_name {exp_name} --hyper_search 1 --retrain {retrain} "
        for k, v in chosen_dict.items():
            command += f"--{k} {v} "

        if debug:
            command += "--debug 1 "

        current_metric = run_and_get_metric(
            command, config, chosen_dict, monitor_metric, debug, args
        )
        evaluated_cache[key] = current_metric
        return current_metric

    with open(log_file, "a") as f:
        for hyper_name, hyper_values in hyper_dict.items():
            if len(hyper_values) == 1:
                best_hyper[hyper_name] = hyper_values[0]
                continue

            print(f"{hyper_name} => {hyper_values}")

            # 先构造“固定参数”：已确定的 best_hyper + 其他未搜索参数的默认值（但不写入 best_hyper）
            fixed_params = dict(best_hyper)
            for other_name, other_values in hyper_dict.items():
                if other_name == hyper_name:
                    continue
                if other_name not in fixed_params:
                    fixed_params[other_name] = other_values[0]

            # baseline（默认取当前超参的第一个值）
            baseline_val = hyper_values[0]
            baseline_dict = dict(fixed_params)
            baseline_dict[hyper_name] = baseline_val

            # 先确保 baseline 有 metric（有缓存就直接用；没有就跑一次）
            baseline_metric = run_once_with_cache(baseline_dict)

            # 初始化本轮最优为 baseline（这样就可以“直接从第二个值开始试”，但仍保留 baseline 作为比较基准）
            local_best_metric = baseline_metric
            current_best_value = baseline_val
            write_and_print(
                f"{hyper_name}: {baseline_val}, Metric: {baseline_metric:5.4f} (baseline)"
            )

            # 直接从第二个值开始（避免重复）
            for value in hyper_values[1:]:
                chosen_dict = dict(fixed_params)
                chosen_dict[hyper_name] = value

                current_metric = run_once_with_cache(chosen_dict)

                if reverse:
                    if current_metric > local_best_metric:
                        local_best_metric = current_metric
                        current_best_value = value
                else:
                    if current_metric < local_best_metric:
                        local_best_metric = current_metric
                        current_best_value = value

                write_and_print(f"{hyper_name}: {value}, Metric: {current_metric:5.4f}")

            # 更新该超参的最优值
            best_hyper[hyper_name] = current_best_value
            write_and_print(
                f"==> Best {hyper_name}: {current_best_value}, local_best_metric: {local_best_metric:5.4f}\n"
            )

        write_and_print(f"The Best Hyperparameters: {best_hyper}\n")

    return best_hyper


def grid_search_hyperparameters(
    exp_name, hyper_dict, retrain, monitor_metric, reverse, debug, args
):
    """
    进行网格搜索（笛卡尔积搜索所有超参数组合）
    """
    config = get_config(exp_name)

    log_file = f"./run.log"
    hyper_keys = list(hyper_dict.keys())
    hyper_values_list = [hyper_dict[k] for k in hyper_keys]

    best_metric = 0 if reverse else 1e9
    best_combo = None

    with open(log_file, "a") as f:
        f.write("\n=== Grid Search ===\n")
        for combo in product(*hyper_values_list):
            # combo 是一个元组，如 (10, 0.1) -> 对应 (Rank=10, Order=0.1)
            combo_dict = dict(zip(hyper_keys, combo))

            # 构建命令
            command = f"python run_train.py --exp_name {exp_name} --hyper_search 1 --retrain {retrain} "
            # 在命令里添加所有超参数
            for param_key, param_val in combo_dict.items():
                command += f"--{param_key} {param_val} "

            # 先给其他未出现在 combo_dict 的超参数，指定其默认值
            for other_key, other_values in hyper_dict.items():
                if other_key not in combo_dict:
                    command += f"--{other_key} {other_values[0]} "

            f.write(f"COMMAND: {command}\n")
            # 运行并获取结果
            current_metric = run_and_get_metric(
                command, config, combo_dict, monitor_metric, debug, args
            )

            if reverse:
                # 分类，metric 越大越好
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_combo = combo_dict
            else:
                # 回归/预测，metric (MAE) 越小越好
                if current_metric < best_metric:
                    best_metric = current_metric
                    best_combo = combo_dict
            write_and_print(f"Combo: {combo_dict}, Metric= {current_metric}\n")

        # 记录最优组合
        write_and_print(f"Best combo: {best_combo}, Best metric: {best_metric}\n")
    return best_combo


def run_command(command, log_file="./run.log", retry_count=0):
    success = False
    while not success:
        # 获取当前时间并格式化
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 如果是重试的命令，标记为 "Retrying"
        if retry_count > 0:
            retry_message = "Retrying"
        else:
            retry_message = "Running"

        # 将执行的命令和时间写入日志文件

        write_and_print(f"{retry_message} at {current_time}: {command}\n")

        # 直接执行命令，将输出和错误信息打印到终端
        process = subprocess.run(
            f"ulimit -s unlimited; ulimit -c unlimited&& ulimit -a && echo {command} &&"
            + command,
            shell=True,
        )

        # 根据返回码判断命令是否成功执行
        if process.returncode == 0:
            success = True
        else:
            with open(log_file, "a") as f:
                f.write(f"Command failed, retrying in 3 seconds: {command}\n")
            retry_count += 1
            time.sleep(3)  # 等待一段时间后重试
