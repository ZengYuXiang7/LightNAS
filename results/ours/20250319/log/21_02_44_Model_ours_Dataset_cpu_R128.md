```python
|2025-03-19 21:02:44| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'dataset': cpu, 'debug': False, 'decay': 0.0001,
     'density': 0.01, 'device': cuda, 'epochs': 500, 'eval_set': True,
     'ffn_method': moe, 'hyper_search': False, 'idx': 0, 'log': <utils.exp_logger.Logger object at 0x76ddacca0050>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': ours,
     'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'rank': 128,
     'record': True, 'retrain': True, 'rounds': 1, 'seed': 0,
     'shuffle': True, 'train_size': 100, 'try_exp': 1, 'use_train_size': True,
     'verbose': 10,
}
|2025-03-19 21:02:44| ********************Experiment Start********************
|2025-03-19 21:04:00| Round=1 BestEpoch=144 MAE=0.0003 RMSE=0.0004 NMAE=0.0559 NRMSE=0.0782 Training_time=17.1 s 
|2025-03-19 21:04:00| ********************Experiment Results:********************
|2025-03-19 21:04:00| NMAE: 0.0559 ± 0.0000
|2025-03-19 21:04:00| NRMSE: 0.0782 ± 0.0000
|2025-03-19 21:04:00| MAE: 0.0003 ± 0.0000
|2025-03-19 21:04:00| RMSE: 0.0004 ± 0.0000
|2025-03-19 21:04:00| Acc_10: 0.8360 ± 0.0000
|2025-03-19 21:04:00| train_time: 17.0664 ± 0.0000
|2025-03-19 21:04:01| Flops: 152178688
|2025-03-19 21:04:01| Params: 528001
|2025-03-19 21:04:01| Inference time: 1.92 ms
|2025-03-19 21:04:01| ********************Experiment Success********************
```

