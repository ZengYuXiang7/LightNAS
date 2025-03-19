```python
|2025-03-19 20:17:45| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'data_scaler': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.02, 'device': cuda, 'epochs': 200,
     'eval_set': True, 'ffn_method': moe, 'hyper_search': False, 'idx': 0,
     'log': <utils.exp_logger.Logger object at 0x7a256b9621b0>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'pred_len': 96,
     'rank': 128, 'record': True, 'retrain': True, 'rounds': 1,
     'seed': 0, 'seq_len': 96, 'shuffle': False, 'train_size': 500,
     'try_exp': 1, 'ts_var': 0, 'verbose': 10,
}
|2025-03-19 20:17:45| ********************Experiment Start********************
|2025-03-19 20:18:40| Round=1 BestEpoch= 96 MAE=0.0003 RMSE=0.0005 NMAE=0.0667 NRMSE=0.0908 Training_time=12.7 s 
|2025-03-19 20:18:40| ********************Experiment Results:********************
|2025-03-19 20:18:40| NMAE: 0.0667 ± 0.0000
|2025-03-19 20:18:40| NRMSE: 0.0908 ± 0.0000
|2025-03-19 20:18:40| MAE: 0.0003 ± 0.0000
|2025-03-19 20:18:40| RMSE: 0.0005 ± 0.0000
|2025-03-19 20:18:40| Acc_10: 0.7405 ± 0.0000
|2025-03-19 20:18:40| train_time: 12.6550 ± 0.0000
|2025-03-19 20:18:40| Flops: 57069568
|2025-03-19 20:18:40| Params: 197505
|2025-03-19 20:18:40| Inference time: 1.39 ms
|2025-03-19 20:18:40| ********************Experiment Success********************
```

