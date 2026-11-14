"""执行 GBDT 回归并保存逐轮实验结果。"""

import pickle

from sklearn.metrics import mean_squared_error

from data_provider.ml_data import load_ml_data
from models.ml import fit_ml_model
from utils.metrics import ErrorMetrics
from utils.observability import result_checkpoint_path
from utils.result_contract import build_data_config, publish_result
from utils.utils import set_seed


def run_ml_experiments(args, log, result_name):
    # 每轮按种子划分数据，保留原有验证集选参和重新拟合流程。
    rounds = []
    for run_id in range(args.rounds):
        set_seed(args.seed + run_id)
        train_x, train_y, valid_x, valid_y, test_x, test_y = load_ml_data(args)
        data_config = build_data_config(args, len(test_y), len(train_y), len(valid_y), len(test_y))
        log(f'Round {run_id + 1}: train={len(train_y)} valid={len(valid_y)} test={len(test_y)}')
        model, parameters, best_score = fit_ml_model(train_x, train_y, valid_x, valid_y)
        refit_valid_mse = float(mean_squared_error(valid_y, model.predict(valid_x)))
        prediction = model.predict(test_x)
        metrics = ErrorMetrics(test_y, prediction)
        model_name = type(model).__name__
        checkpoint = result_checkpoint_path(args.dataset_name, result_name, run_id + 1)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open('wb') as file:
            pickle.dump(model, file)

        # 记录真实评估模型、选中的参数及其验证分数。
        test_metrics = {}
        for metric, value in metrics.items():
            test_metrics[metric] = float(value)
        round_record = {
            'algorithm': 'gradient_boosting',
            'seed': args.seed + run_id,
            'best_epoch': 0,
            'best_valid': {'MSE': refit_valid_mse},
            'test_metrics': test_metrics,
            'checkpoint': checkpoint.as_posix(),
            'checkpoint_format': 'pickle',
            'selected_parameters': parameters,
            'model_parameters': model.get_params(deep=False),
            'selection_validation_mse': best_score,
            'early_stop': None,
        }
        rounds.append(round_record)
        log(f'{model_name} best validation parameters: {parameters}')
        values = ' '.join(f'{name}={value:.4f}' for name, value in metrics.items())
        log(f'{model_name} test {values}')

        # 每轮更新记录；最终轮追加批次摘要。
        if args.record:
            publish_result(args, result_name, model_name, data_config, rounds, run_id == args.rounds - 1)
    return rounds
