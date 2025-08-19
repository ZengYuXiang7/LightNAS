import os
import pickle
import numpy as np
from data_provider.data_control import load_data
from data_provider.data_scaler import get_scaler
from utils.exp_config import get_config
from utils.utils import set_seed

# ===== 1. 获取配置 =====
config = get_config('TransModelConfig')
for runid in range(5):
    config.runid = runid
    set_seed(config.seed + runid)

    # 数据集名和存储目录
    dataset_name = os.path.splitext(os.path.basename(config.dst_dataset))[0]
    save_dir = os.path.join("data", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # 生成当前 run 的文件名
    def file_path(var_name):
        """生成形如 dataset_x_run1.pkl 的文件路径"""
        return os.path.join(save_dir, f"{dataset_name}_{var_name}_round{config.runid}.pkl")

    # 需要的文件名列表
    expected_files = [
        file_path("train_x"), file_path("valid_x"), file_path("test_x"),
        file_path("train_y"), file_path("valid_y"), file_path("test_y")
    ]

    # ===== 2. 如果已经存在所有文件，则直接读取 =====
    if all(os.path.exists(f) for f in expected_files):
        def load_pkl(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        train_x = load_pkl(file_path("train_x"))
        valid_x = load_pkl(file_path("valid_x"))
        test_x  = load_pkl(file_path("test_x"))
        train_y = load_pkl(file_path("train_y"))
        valid_y = load_pkl(file_path("valid_y"))
        test_y  = load_pkl(file_path("test_y"))

        print(f"✅ Loaded preprocessed data from {save_dir}")

    else:
        # ===== 3. 加载并转为 float32 =====
        x, y = load_data(config)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # ===== 4. 切分比例 =====
        parts = [int(s) for s in config.spliter_ratio.strip().split(':')]
        total = sum(parts)
        train_ratio, valid_ratio = parts[0] / total, parts[1] / total

        # ===== 5. 打乱切分 =====
        n = len(x)
        train_size = int(n * train_ratio)
        valid_size = int(n * valid_ratio) if config.eval_set else 0

        indices = np.random.permutation(n)
        train_idx = indices[:train_size]
        valid_idx = indices[train_size:train_size + valid_size]
        test_idx  = indices[train_size + valid_size:]

        train_x, train_y = x[train_idx], y[train_idx]
        valid_x, valid_y = x[valid_idx], y[valid_idx]
        test_x,  test_y  = x[test_idx],  y[test_idx]

        # ===== 6. 归一化 =====
        x_scaler = get_scaler(train_x, config, 'None')
        y_scaler = get_scaler(train_y, config, 'globalminmax')

        train_x = x_scaler.transform(train_x)
        valid_x = x_scaler.transform(valid_x)
        test_x  = x_scaler.transform(test_x)

        train_y = y_scaler.transform(train_y).astype(np.float32)
        valid_y = y_scaler.transform(valid_y).astype(np.float32)
        test_y  = y_scaler.transform(test_y).astype(np.float32)

        # ===== 7. 保存每个变量 =====
        def save_pkl(var_name, data):
            with open(file_path(var_name), 'wb') as f:
                pickle.dump(data, f)

        save_pkl("train_x", train_x)
        save_pkl("valid_x", valid_x)
        save_pkl("test_x",  test_x)
        save_pkl("train_y", train_y)
        save_pkl("valid_y", valid_y)
        save_pkl("test_y",  test_y)

        print(f"✅ Data processed and saved to {save_dir}")