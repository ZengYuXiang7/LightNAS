```python
|2025-03-19 20:40:13| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'dataset': cpu, 'debug': False, 'decay': 0.0001,
     'density': 0.01, 'device': cuda, 'epochs': 500, 'eval_set': True,
     'ffn_method': moe, 'hyper_search': False, 'idx': 0, 'log': <utils.exp_logger.Logger object at 0x727ec65949e0>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': ours,
     'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'rank': 128,
     'record': True, 'retrain': True, 'rounds': 1, 'seed': 0,
     'shuffle': True, 'train_size': 100, 'try_exp': 1, 'use_train_size': True,
     'verbose': 10,
}
|2025-03-19 20:40:13| ********************Experiment Start********************
|2025-03-19 20:41:32| Round=1 BestEpoch=162 MAE=0.0003 RMSE=0.0004 NMAE=0.0526 NRMSE=0.0726 Training_time=18.4 s 
|2025-03-19 20:41:32| ********************Experiment Results:********************
|2025-03-19 20:41:32| NMAE: 0.0526 ± 0.0000
|2025-03-19 20:41:32| NRMSE: 0.0726 ± 0.0000
|2025-03-19 20:41:32| MAE: 0.0003 ± 0.0000
|2025-03-19 20:41:32| RMSE: 0.0004 ± 0.0000
|2025-03-19 20:41:32| Acc_10: 0.8549 ± 0.0000
|2025-03-19 20:41:32| train_time: 18.3501 ± 0.0000
|2025-03-19 20:41:32| Flops: 76091392
|2025-03-19 20:41:32| Params: 263297
|2025-03-19 20:41:32| Inference time: 1.80 ms
|2025-03-19 20:41:32| ********************Experiment Success********************
```

