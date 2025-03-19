```python
|2025-03-19 20:25:05| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'data_scaler': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.02, 'device': cuda, 'epochs': 500,
     'eval_set': True, 'ffn_method': moe, 'hyper_search': False, 'idx': 0,
     'log': <utils.exp_logger.Logger object at 0x7556fcdf8bf0>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms,
     'num_layers': 32, 'optim': AdamW, 'path': ./datasets/, 'patience': 50,
     'pred_len': 96, 'rank': 128, 'record': True, 'retrain': True,
     'rounds': 1, 'seed': 0, 'seq_len': 96, 'shuffle': False,
     'train_size': 500, 'try_exp': 1, 'ts_var': 0, 'verbose': 10,
}
|2025-03-19 20:25:05| ********************Experiment Start********************
|2025-03-19 20:26:29| Round=1 BestEpoch=159 MAE=0.0004 RMSE=0.0005 NMAE=0.0809 NRMSE=0.1040 Training_time=23.0 s 
|2025-03-19 20:26:29| ********************Experiment Results:********************
|2025-03-19 20:26:29| NMAE: 0.0809 ± 0.0000
|2025-03-19 20:26:29| NRMSE: 0.1040 ± 0.0000
|2025-03-19 20:26:29| MAE: 0.0004 ± 0.0000
|2025-03-19 20:26:29| RMSE: 0.0005 ± 0.0000
|2025-03-19 20:26:29| Acc_10: 0.6945 ± 0.0000
|2025-03-19 20:26:29| train_time: 23.0271 ± 0.0000
|2025-03-19 20:26:30| Flops: 76091392
|2025-03-19 20:26:30| Params: 263297
|2025-03-19 20:26:30| Inference time: 1.81 ms
|2025-03-19 20:26:30| ********************Experiment Success********************
```

