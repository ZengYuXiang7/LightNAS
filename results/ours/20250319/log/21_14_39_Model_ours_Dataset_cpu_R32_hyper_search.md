```python
|2025-03-19 21:14:39| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'dataset': cpu, 'debug': False, 'decay': 0.0001,
     'density': 0.01, 'device': cuda, 'epochs': 500, 'eval_set': True,
     'ffn_method': moe, 'hyper_search': True, 'idx': 0, 'log': <utils.exp_logger.Logger object at 0x7b6ae11b44a0>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': ours,
     'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'rank': 32,
     'record': True, 'retrain': True, 'rounds': 1, 'seed': 0,
     'shuffle': True, 'train_size': 100, 'try_exp': 1, 'use_train_size': True,
     'verbose': 10,
}
|2025-03-19 21:14:39| ********************Experiment Start********************
|2025-03-19 21:18:51| Round=1 BestEpoch=179 MAE=0.0002 RMSE=0.0003 NMAE=0.0455 NRMSE=0.0646 Training_time=36.9 s 
|2025-03-19 21:18:51| ********************Experiment Results:********************
|2025-03-19 21:18:51| NMAE: 0.0455 ± 0.0000
|2025-03-19 21:18:51| NRMSE: 0.0646 ± 0.0000
|2025-03-19 21:18:51| MAE: 0.0002 ± 0.0000
|2025-03-19 21:18:51| RMSE: 0.0003 ± 0.0000
|2025-03-19 21:18:51| Acc_10: 0.8836 ± 0.0000
|2025-03-19 21:18:51| train_time: 36.9400 ± 0.0000
|2025-03-19 21:18:53| Flops: 77857792
|2025-03-19 21:18:53| Params: 269345
|2025-03-19 21:18:53| Inference time: 15.53 ms
|2025-03-19 21:18:54| ********************Experiment Success********************
```

