```python
|2025-03-19 21:06:34| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'dataset': cpu, 'debug': False, 'decay': 0.0001,
     'density': 0.01, 'device': cuda, 'epochs': 500, 'eval_set': True,
     'ffn_method': moe, 'hyper_search': False, 'idx': 0, 'log': <utils.exp_logger.Logger object at 0x72d4590344a0>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': ours,
     'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'rank': 128,
     'record': True, 'retrain': True, 'rounds': 1, 'seed': 0,
     'shuffle': True, 'train_size': 100, 'try_exp': 1, 'use_train_size': True,
     'verbose': 10,
}
|2025-03-19 21:06:34| ********************Experiment Start********************
|2025-03-19 21:07:33| Round=1 BestEpoch=128 MAE=0.0004 RMSE=0.0009 NMAE=0.0916 NRMSE=0.1714 Training_time=14.0 s 
|2025-03-19 21:07:33| ********************Experiment Results:********************
|2025-03-19 21:07:33| NMAE: 0.0916 ± 0.0000
|2025-03-19 21:07:33| NRMSE: 0.1714 ± 0.0000
|2025-03-19 21:07:33| MAE: 0.0004 ± 0.0000
|2025-03-19 21:07:33| RMSE: 0.0009 ± 0.0000
|2025-03-19 21:07:33| Acc_10: 0.7544 ± 0.0000
|2025-03-19 21:07:33| train_time: 13.9777 ± 0.0000
|2025-03-19 21:07:33| Flops: 133894144
|2025-03-19 21:07:33| Params: 465025
|2025-03-19 21:07:33| Inference time: 1.02 ms
|2025-03-19 21:07:34| ********************Experiment Success********************
```

