"""组织单设备延迟实验的契约记录和批次摘要。"""

from pathlib import Path
from time import strftime

from utils.observability import CONTRACT_VERSION, aggregate_test_metrics, build_dataset_significance
from utils.observability import result_record_path, result_report_path, write_result_record, write_result_report


def build_data_config(args, available_count, train_count, valid_count, test_count):
    # 记录训练、验证及包含全部样本的测试范围。
    source = args.latency_table
    if args.model == 'ml':
        split = 'Random permutation; fixed validation set after training set; test includes all samples'
        target_transform = '原始秒值'
    else:
        split = 'Random permutation; 10% of all samples for validation after training set; test includes all samples'
        target_transform = '全体样本最大值归一化；评价时还原到秒'
    seeds = []
    for run_id in range(args.rounds):
        seeds.append(args.seed + run_id)
    data_config = {
        'source': source,
        'available_samples': available_count,
        'split': split,
        'split_seeds': seeds,
        'train_samples': train_count,
        'valid_samples': valid_count,
        'test_samples': test_count,
        'features': '八节点操作编码；图模型按需加入全局节点',
        'feature_normalization': '无',
        'target': '单设备延迟，单位秒',
        'target_transform': target_transform,
    }
    return data_config


def build_model_config(args, model_name):
    # 记录真实模型身份和选定模型使用的结构参数。
    model_config = {'model': model_name}
    if args.model == 'ml':
        model_config['parameters'] = '各轮实际参数见 rounds.model_parameters'
        model_config['selection'] = '原参数网格，以验证集 MSE 选择后重新拟合'
    else:
        model_config['rank'] = args.rank
        if args.model == 'ours':
            model_config['op_encoder'] = args.op_encoder
            model_config['graph_encoder'] = args.graph_encoder
            model_config['order'] = args.order
            model_config['heads'] = args.heads
            model_config['Ablation'] = args.Ablation
            model_config['device_name'] = args.device_name
            model_config['dropout'] = 0.1
        elif args.model == 'brp_nas':
            model_config['layers'] = 6
            model_config['dropout'] = 0.1
        elif args.model == 'gcn':
            model_config['layers'] = 6
            model_config['dropout'] = 0.1
        elif args.model == 'mlp':
            model_config['hidden_layers'] = f'{args.rank}, {args.rank // 2}'
        elif args.model in ('lstm', 'gru', 'birnn'):
            model_config['layers'] = 1
            model_config['bidirectional'] = args.model == 'birnn'
        elif args.model == 'dnnperf':
            model_config['node_features'] = 7
            model_config['edge_features'] = 2
            model_config['node_hidden'] = args.rank
            model_config['edge_hidden'] = args.rank
        elif args.model == 'flops':
            model_config['flops_table'] = args.latency_table
    return model_config


def build_train_config(args):
    # 训练设置只包含所选分支的实际参数。
    train_config = {'seed': args.seed, 'rounds': args.rounds, 'train_size': args.train_size}
    if args.model == 'ml':
        train_config['valid_size'] = args.ml_valid_size
        train_config['selection_metric'] = 'MSE'
    else:
        train_config['epochs'] = args.epochs
        train_config['patience'] = args.patience
        train_config['batch_size'] = args.bs
        train_config['device'] = args.device
        train_config['learning_rate'] = args.lr
        train_config['optimizer'] = args.optim
        train_config['weight_decay'] = args.decay
        train_config['loss'] = args.loss_func
    return train_config


def publish_result(args, result_name, model_name, data_config, rounds, append_summary):
    # 相同条件才允许比较；不同验证集或标签变换不会混作配对实验。
    condition = {
        'domain': args.train_device,
        'hardware_type': args.dataset,
        'source': data_config['source'],
        'train_samples': data_config['train_samples'],
        'valid_samples': data_config['valid_samples'],
        'test_samples': data_config['test_samples'],
        'target_transform': data_config['target_transform'],
        'seed': args.seed,
        'rounds': args.rounds,
    }
    result = {
        'contract_version': CONTRACT_VERSION,
        'dataset': args.dataset_name,
        'result_name': result_name,
        'model_name': model_name,
        'purpose': args.purpose,
        'condition': condition,
        'model_config': build_model_config(args, model_name),
        'train_config': build_train_config(args),
        'data_config': data_config,
        'mean_std': aggregate_test_metrics(rounds),
        'rounds': rounds,
    }
    batch_dir = Path(args.batch_dir)
    result['significance'] = build_dataset_significance(args.dataset_name, result_name, result, batch_dir / 'records')
    record_path = result_record_path(args.dataset_name, result_name, batch_dir)
    report_path = result_report_path(args.dataset_name, result_name, batch_dir)
    write_result_record(record_path, result)
    write_result_report(report_path, result)

    # 批次日志只追加已完成配置的汇总。
    if append_summary:
        values = []
        for name, summary in result['mean_std'].items():
            values.append(f"{name}={summary['mean']:.4f}±{summary['std']:.4f}")
        batch_dir.mkdir(parents=True, exist_ok=True)
        with (batch_dir / 'run.log').open('a', encoding='utf-8') as file:
            file.write(f"{strftime('%Y-%m-%d %H:%M:%S')} {model_name} {result_name} {' '.join(values)}\n")
    return result
