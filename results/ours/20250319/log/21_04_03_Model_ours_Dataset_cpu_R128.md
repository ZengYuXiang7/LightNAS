```python
|2025-03-19 21:04:03| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'dataset': cpu, 'debug': False, 'decay': 0.0001,
     'density': 0.01, 'device': cuda, 'epochs': 500, 'eval_set': True,
     'ffn_method': moe, 'hyper_search': False, 'idx': 0, 'log': <utils.exp_logger.Logger object at 0x77fac5fb9790>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': ours,
     'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'rank': 128,
     'record': True, 'retrain': True, 'rounds': 1, 'seed': 0,
     'shuffle': True, 'train_size': 100, 'try_exp': 1, 'use_train_size': True,
     'verbose': 10,
}
|2025-03-19 21:04:03| ********************Experiment Start********************
|2025-03-19 21:05:28| Round=1 BestEpoch=162 MAE=0.0002 RMSE=0.0003 NMAE=0.0379 NRMSE=0.0505 Training_time=20.2 s 
|2025-03-19 21:05:28| ********************Experiment Results:********************
|2025-03-19 21:05:28| NMAE: 0.0379 ± 0.0000
|2025-03-19 21:05:28| NRMSE: 0.0505 ± 0.0000
|2025-03-19 21:05:28| MAE: 0.0002 ± 0.0000
|2025-03-19 21:05:28| RMSE: 0.0003 ± 0.0000
|2025-03-19 21:05:28| Acc_10: 0.9379 ± 0.0000
|2025-03-19 21:05:28| train_time: 20.1837 ± 0.0000
|2025-03-19 21:05:29| Flops: 152178688
|2025-03-19 21:05:29| Params: 528001
|2025-03-19 21:05:29| Inference time: 2.08 ms
|2025-03-19 21:05:29| ********************Experiment Success********************
```

