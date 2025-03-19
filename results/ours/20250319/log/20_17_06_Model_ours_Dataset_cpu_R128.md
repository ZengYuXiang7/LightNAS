```python
|2025-03-19 20:17:06| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'data_scaler': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.02, 'device': cuda, 'epochs': 200,
     'eval_set': True, 'ffn_method': moe, 'hyper_search': False, 'idx': 0,
     'log': <utils.exp_logger.Logger object at 0x751206290cb0>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'pred_len': 96,
     'rank': 128, 'record': True, 'retrain': True, 'rounds': 1,
     'seed': 0, 'seq_len': 96, 'shuffle': False, 'train_size': 500,
     'try_exp': 1, 'ts_var': 0, 'verbose': 10,
}
|2025-03-19 20:17:06| ********************Experiment Start********************
|2025-03-19 20:17:29| Round=1 BestEpoch= 21 MAE=0.0007 RMSE=0.0009 NMAE=0.1392 NRMSE=0.1677 Training_time=2.5 s 
|2025-03-19 20:17:29| ********************Experiment Results:********************
|2025-03-19 20:17:29| NMAE: 0.1392 ± 0.0000
|2025-03-19 20:17:29| NRMSE: 0.1677 ± 0.0000
|2025-03-19 20:17:29| MAE: 0.0007 ± 0.0000
|2025-03-19 20:17:29| RMSE: 0.0009 ± 0.0000
|2025-03-19 20:17:29| Acc_10: 0.4348 ± 0.0000
|2025-03-19 20:17:29| train_time: 2.4538 ± 0.0000
|2025-03-19 20:17:29| Flops: 14454784
|2025-03-19 20:17:29| Params: 50177
|2025-03-19 20:17:29| Inference time: 0.32 ms
|2025-03-19 20:17:30| ********************Experiment Success********************
```

