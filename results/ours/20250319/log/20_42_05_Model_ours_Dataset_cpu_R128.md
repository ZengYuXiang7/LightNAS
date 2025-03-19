```python
|2025-03-19 20:42:05| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'dataset': cpu, 'debug': False, 'decay': 0.0001,
     'density': 0.01, 'device': cuda, 'epochs': 500, 'eval_set': True,
     'ffn_method': moe, 'hyper_search': False, 'idx': 0, 'log': <utils.exp_logger.Logger object at 0x71fa6b1449e0>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': ours,
     'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'rank': 128,
     'record': True, 'retrain': True, 'rounds': 1, 'seed': 0,
     'shuffle': True, 'train_size': 100, 'try_exp': 1, 'use_train_size': True,
     'verbose': 10,
}
|2025-03-19 20:42:05| ********************Experiment Start********************
|2025-03-19 20:43:36| Round=1 BestEpoch=196 MAE=0.0002 RMSE=0.0003 NMAE=0.0408 NRMSE=0.0571 Training_time=22.0 s 
|2025-03-19 20:43:36| ********************Experiment Results:********************
|2025-03-19 20:43:36| NMAE: 0.0408 ± 0.0000
|2025-03-19 20:43:36| NRMSE: 0.0571 ± 0.0000
|2025-03-19 20:43:36| MAE: 0.0002 ± 0.0000
|2025-03-19 20:43:36| RMSE: 0.0003 ± 0.0000
|2025-03-19 20:43:36| Acc_10: 0.8840 ± 0.0000
|2025-03-19 20:43:36| train_time: 22.0323 ± 0.0000
|2025-03-19 20:43:36| Flops: 76091392
|2025-03-19 20:43:36| Params: 263297
|2025-03-19 20:43:36| Inference time: 1.76 ms
|2025-03-19 20:43:37| ********************Experiment Success********************
```

