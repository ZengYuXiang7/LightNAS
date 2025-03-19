```python
|2025-03-19 21:00:18| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'dataset': cpu, 'debug': False, 'decay': 0.0001,
     'density': 0.01, 'device': cuda, 'epochs': 500, 'eval_set': True,
     'ffn_method': moe, 'hyper_search': False, 'idx': 0, 'log': <utils.exp_logger.Logger object at 0x72fb4b90cb00>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': ours,
     'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'rank': 128,
     'record': True, 'retrain': True, 'rounds': 1, 'seed': 0,
     'shuffle': True, 'train_size': 100, 'try_exp': 1, 'use_train_size': True,
     'verbose': 10,
}
|2025-03-19 21:00:18| ********************Experiment Start********************
|2025-03-19 21:02:21| Round=1 BestEpoch=262 MAE=0.0002 RMSE=0.0004 NMAE=0.0504 NRMSE=0.0731 Training_time=31.5 s 
|2025-03-19 21:02:21| ********************Experiment Results:********************
|2025-03-19 21:02:21| NMAE: 0.0504 ± 0.0000
|2025-03-19 21:02:21| NRMSE: 0.0731 ± 0.0000
|2025-03-19 21:02:21| MAE: 0.0002 ± 0.0000
|2025-03-19 21:02:21| RMSE: 0.0004 ± 0.0000
|2025-03-19 21:02:21| Acc_10: 0.8729 ± 0.0000
|2025-03-19 21:02:21| train_time: 31.5180 ± 0.0000
|2025-03-19 21:02:21| Flops: 150999040
|2025-03-19 21:02:21| Params: 525953
|2025-03-19 21:02:21| Inference time: 2.10 ms
|2025-03-19 21:02:22| ********************Experiment Success********************
```

