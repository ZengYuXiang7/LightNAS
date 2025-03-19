```python
|2025-03-19 21:18:55| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'dataset': cpu, 'debug': False, 'decay': 0.0001,
     'density': 0.01, 'device': cuda, 'epochs': 500, 'eval_set': True,
     'ffn_method': moe, 'hyper_search': True, 'idx': 0, 'log': <utils.exp_logger.Logger object at 0x7e3d4bb949e0>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': ours,
     'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'rank': 64,
     'record': True, 'retrain': True, 'rounds': 1, 'seed': 0,
     'shuffle': True, 'train_size': 100, 'try_exp': 1, 'use_train_size': True,
     'verbose': 10,
}
|2025-03-19 21:18:55| ********************Experiment Start********************
|2025-03-19 21:21:33| Round=1 BestEpoch= 86 MAE=0.0002 RMSE=0.0003 NMAE=0.0494 NRMSE=0.0651 Training_time=18.6 s 
|2025-03-19 21:21:33| ********************Experiment Results:********************
|2025-03-19 21:21:33| NMAE: 0.0494 ± 0.0000
|2025-03-19 21:21:33| NRMSE: 0.0651 ± 0.0000
|2025-03-19 21:21:33| MAE: 0.0002 ± 0.0000
|2025-03-19 21:21:33| RMSE: 0.0003 ± 0.0000
|2025-03-19 21:21:33| Acc_10: 0.8749 ± 0.0000
|2025-03-19 21:21:33| train_time: 18.6154 ± 0.0000
|2025-03-19 21:21:34| Flops: 306710528
|2025-03-19 21:21:34| Params: 1062977
|2025-03-19 21:21:34| Inference time: 15.99 ms
|2025-03-19 21:21:35| ********************Experiment Success********************
```

