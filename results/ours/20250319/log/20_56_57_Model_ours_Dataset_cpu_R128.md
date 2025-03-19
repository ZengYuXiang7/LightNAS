```python
|2025-03-19 20:56:57| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'dataset': cpu, 'debug': False, 'decay': 0.0001,
     'density': 0.01, 'device': cuda, 'epochs': 500, 'eval_set': True,
     'ffn_method': moe, 'hyper_search': False, 'idx': 0, 'log': <utils.exp_logger.Logger object at 0x77ec3be8d4c0>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': ours,
     'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'rank': 128,
     'record': True, 'retrain': True, 'rounds': 1, 'seed': 0,
     'shuffle': True, 'train_size': 100, 'try_exp': 1, 'use_train_size': True,
     'verbose': 10,
}
|2025-03-19 20:56:57| ********************Experiment Start********************
|2025-03-19 20:58:25| Round=1 BestEpoch=178 MAE=0.0002 RMSE=0.0003 NMAE=0.0438 NRMSE=0.0615 Training_time=21.2 s 
|2025-03-19 20:58:25| ********************Experiment Results:********************
|2025-03-19 20:58:25| NMAE: 0.0438 ± 0.0000
|2025-03-19 20:58:25| NRMSE: 0.0615 ± 0.0000
|2025-03-19 20:58:25| MAE: 0.0002 ± 0.0000
|2025-03-19 20:58:25| RMSE: 0.0003 ± 0.0000
|2025-03-19 20:58:25| Acc_10: 0.9057 ± 0.0000
|2025-03-19 20:58:25| train_time: 21.1750 ± 0.0000
|2025-03-19 20:58:26| Flops: 152178688
|2025-03-19 20:58:26| Params: 528001
|2025-03-19 20:58:26| Inference time: 1.94 ms
|2025-03-19 20:58:26| ********************Experiment Success********************
```

