"""解析数据、训练和所选模型的命令行参数。"""

import argparse


def parse_bool(value):
    # 保持原命令支持的 0/1 和 true/false 写法。
    value = value.lower()
    if value in ('1', 'true'):
        result = True
    elif value in ('0', 'false'):
        result = False
    else:
        raise argparse.ArgumentTypeError('Use 0/1 or true/false')
    return result


def get_config(argv=None):
    # 先确定模型，再注册该模型实际使用的结构参数。
    models = ('ours', 'gcn', 'brp_nas', 'mlp', 'lstm', 'gru', 'birnn', 'dnnperf', 'flops', 'ml')
    initial = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    initial.add_argument('--model', choices=models, default='mlp')
    selected, _ = initial.parse_known_args(argv)
    parser = argparse.ArgumentParser(description='NAS-Bench-201 latency prediction', allow_abbrev=False)
    parser.add_argument('--model', choices=models, default=selected.model)

    # 公共数据、训练和结果参数沿用项目默认值。
    parser.add_argument('--dataset', default='cpu')
    parser.add_argument('--dataset_name', default='nasbench201')
    parser.add_argument('--batch_dir', default='run_exp/manual')
    parser.add_argument('--purpose', default='Single-device latency prediction')
    parser.add_argument('--latency_table', default='./data/nasbench201_complete.pickle')
    parser.add_argument('--train_device', default='desktop-cpu-core-i7-7820x')
    parser.add_argument('--train_size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--debug', type=parse_bool, default=False)
    parser.add_argument('--record', type=parse_bool, default=True)
    parser.add_argument('--result_name', default='')

    # 保留原配置中不同模型的结构、学习率和批大小默认值。
    lr_default = 0.001
    batch_default = 32
    if selected.model == 'ours':
        rank_default = 100
        parser.add_argument('--op_encoder', choices=('embed', 'value', 'one_hot'), default='embed')
        parser.add_argument('--graph_encoder', choices=('gcn', 'graphsage', 'gat'), default='gat')
        parser.add_argument('--order', type=int, default=4)
        parser.add_argument('--heads', type=int, default=8)
        parser.add_argument('--Ablation', type=int, default=0)
        parser.add_argument('--device_name', default='core-i7-7820x')
    elif selected.model in ('gcn', 'mlp', 'lstm', 'gru', 'birnn'):
        rank_default = 300
    elif selected.model == 'brp_nas':
        rank_default = 600
        batch_default = 1
    elif selected.model == 'flops':
        rank_default = 32
    elif selected.model == 'dnnperf':
        rank_default = 32
        batch_default = 64
        lr_default = 0.0001
    elif selected.model == 'ml':
        # GBDT 使用固定大小的验证集选择参数。
        parser.add_argument('--ml_valid_size', type=int, default=100)
    if selected.model != 'ml':
        # 神经网络训练与结构参数不传给机器学习分支。
        parser.add_argument('--epochs', type=int, default=3000)
        parser.add_argument('--patience', type=int, default=100)
        parser.add_argument('--print_freq', type=int, default=500)
        parser.add_argument('--tqdm', type=parse_bool, default=True)
        parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--decay', type=float, default=0.0001)
        parser.add_argument('--loss_func', choices=('L1Loss', 'MSELoss', 'SmoothL1Loss'), default='L1Loss')
        parser.add_argument('--optim', default='AdamW')
        parser.add_argument('--rank', type=int, default=rank_default)
        parser.add_argument('--lr', type=float, default=lr_default)
        parser.add_argument('--bs', type=int, default=batch_default)
    args = parser.parse_args(argv)
    return args
