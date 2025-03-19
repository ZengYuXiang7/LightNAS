```python
|2025-03-19 21:07:43| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'dataset': cpu, 'debug': False, 'decay': 0.0001,
     'density': 0.01, 'device': cuda, 'epochs': 500, 'eval_set': True,
     'ffn_method': moe, 'hyper_search': False, 'idx': 0, 'log': <utils.exp_logger.Logger object at 0x7086a12a9610>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': ours,
     'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'rank': 128,
     'record': True, 'retrain': True, 'rounds': 1, 'seed': 0,
     'shuffle': True, 'train_size': 100, 'try_exp': 1, 'use_train_size': True,
     'verbose': 10,
}
|2025-03-19 21:07:43| ********************Experiment Start********************
|2025-03-19 21:08:42| Round=1 BestEpoch=125 MAE=0.0003 RMSE=0.0004 NMAE=0.0602 NRMSE=0.0785 Training_time=13.8 s 
|2025-03-19 21:08:42| ********************Experiment Results:********************
|2025-03-19 21:08:42| NMAE: 0.0602 ± 0.0000
|2025-03-19 21:08:42| NRMSE: 0.0785 ± 0.0000
|2025-03-19 21:08:42| MAE: 0.0003 ± 0.0000
|2025-03-19 21:08:42| RMSE: 0.0004 ± 0.0000
|2025-03-19 21:08:42| Acc_10: 0.8020 ± 0.0000
|2025-03-19 21:08:42| train_time: 13.7852 ± 0.0000
|2025-03-19 21:08:42| Flops: 95555584
|2025-03-19 21:08:42| Params: 331393
|2025-03-19 21:08:42| Inference time: 1.11 ms
|2025-03-19 21:08:43| ********************Experiment Success********************
```

