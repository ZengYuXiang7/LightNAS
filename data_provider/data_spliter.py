import numpy as np
import pickle


def random_split(data, N, tr, vr, seed, config):
    print("用的是随机采样法")
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(N * tr)
    n_valid = int(N * vr)

    return (
        idx[:n_train].tolist(),
        idx[n_train : n_train + n_valid].tolist(),
        idx[n_train + n_valid :].tolist(),
    )


def ours_split(data, N, tr, vr, seed, config):
    print("用的是Kmeans采样法")
    if config.dataset == "201_acc":
        address = "./data/201_traing_sample.pkl"
    elif config.dataset == "101_acc":
        address = "./data/101_traing_sample.pkl"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_idx_all = pickle.load(open(address, "rb"))
    n_train = int(N * tr)

    train_idx = train_idx_all[n_train]

    remaining = list(set(range(N)) - set(train_idx))
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)

    n_valid = int(N * vr)
    valid_idx = remaining[:n_valid]
    test_idx = remaining[n_valid:]

    return train_idx, valid_idx, test_idx


def nnlqp_split(data, N, tr, vr, seed, config):
    print('NNLQP的切分方式')
    train_idx = []
    valid_idx = []
    test_idx = []
    if not config.indomain:
        for i in range(N):
            if data["model_type"][i] != config.test_model_type:
                train_idx.append(i)
            else:
                valid_idx.append(i)
                test_idx.append(i)
    else:
        # 因为数据集是10个模型
        # 1. 建桶：model_type -> indices
        rng = np.random.default_rng(seed)
        model_buckets = {}
        for i, m in enumerate(data["model_type"]):
            model_buckets.setdefault(m, []).append(i)
        train_per_model = 1800
        # 2. 每个桶内切分
        for m, indices in model_buckets.items():
            assert len(indices) >= train_per_model, f"{m} 样本数不足 {train_per_model}"

            indices = indices.copy()
            rng.shuffle(indices)

            train_part = indices[:train_per_model]
            rest = indices[train_per_model:]

            train_idx.extend(train_part)
            valid_idx.extend(rest)
            test_idx.extend(rest)

    return train_idx, valid_idx, test_idx


SPLIT_STRATEGIES = {
    "random": random_split,
    "ours": ours_split,
    # 特定场景的设置
    "nnlqp": nnlqp_split,
}
