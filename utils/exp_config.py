# coding : utf-8
# Author : yuxiang Zeng
import argparse
import importlib.util
import os
import sys
from dataclasses import fields
import glob
import shutil


def get_config(Config='MainConfig'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=Config)
    parser.add_argument('--config_path', type=str, default=f'configs/{Config}.py')
    args, unknown_args = parser.parse_known_args()

    # 动态设置 config_path
    args.config_path = f'configs/{args.exp_name}.py'
    args = load_config(args.config_path, args.exp_name)
    args = update_config_from_args(args, unknown_args)
    
    # 自动清理无效日志和空的 __pycache__ 文件夹
    clear_useless_logs()
    remove_pycache()
    return args


def load_config(file_path, class_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    print(module, class_name)
    config = getattr(module, class_name)()
    return config


def update_config_from_args(config, args):
    it = iter(args)
    for arg in it:
        if arg.startswith("--"):
            if "=" in arg:
                key, value = arg[2:].split("=")
            else:
                key = arg[2:]
                value = next(it)

            field_type = next((f.type for f in fields(config) if f.name == key), str)
            if field_type == bool:
                value = value.lower() in ['true', '1', 'yes']
            else:
                value = field_type(value)
            setattr(config, key, value)
    return config


# 清理无效日志文件
def clear_useless_logs():
    for dirpath, _, _ in os.walk('./results/'):
        if 'log' in dirpath:
            for log_file in glob.glob(os.path.join(dirpath, '*.md')):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        if 'Round=1' not in f.read():
                            os.remove(log_file)
                except Exception as e:
                    print(f"Error processing file {log_file}: {e}")

# 删除空pycache文件夹
def remove_pycache(root_dir="."):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            shutil.rmtree(pycache_path)
    print("✅ All __pycache__ folders removed")
    return True

# 删除空文件夹
def delete_empty_directories(dir_path):
    # 检查目录是否存在
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        # 遍历目录中的所有文件和子目录，从最底层开始
        for root, dirs, files in os.walk(dir_path, topdown=False):
            # 先删除空的子目录
            for name in dirs:
                dir_to_remove = os.path.join(root, name)
                # 如果目录是空的，则删除它
                try:
                    if not os.listdir(dir_to_remove):  # 判断目录是否为空
                        os.rmdir(dir_to_remove)
                        print(f"Directory {dir_to_remove} has been deleted.")
                except FileNotFoundError:
                    # 如果目录已经不存在，忽略此错误
                    pass
            # 检查当前目录是否也是空的，如果是则删除它
            try:
                if not os.listdir(root):  # 判断当前根目录是否为空
                    os.rmdir(root)
                    print(f"Directory {root} has been deleted.")
            except FileNotFoundError:
                # 如果目录已经不存在，忽略此错误
                pass
    else:
        print(f"Directory {dir_path} does not exist.")
