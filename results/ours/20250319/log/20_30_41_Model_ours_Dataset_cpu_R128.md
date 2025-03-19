```python
|2025-03-19 20:30:41| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'data_scaler': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.02, 'device': cuda, 'epochs': 500,
     'eval_set': True, 'ffn_method': moe, 'hyper_search': False, 'idx': 0,
     'log': <utils.exp_logger.Logger object at 0x7d4f31539550>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms,
     'num_layers': 32, 'optim': AdamW, 'path': ./datasets/, 'patience': 50,
     'pred_len': 96, 'rank': 128, 'record': True, 'retrain': True,
     'rounds': 1, 'seed': 0, 'seq_len': 96, 'shuffle': False,
     'train_size': 500, 'try_exp': 1, 'ts_var': 0, 'verbose': 10,
}
|2025-03-19 20:30:41| ********************Experiment Start********************
|2025-03-19 20:31:43| Round=1 BestEpoch= 98 MAE=0.0003 RMSE=0.0004 NMAE=0.0574 NRMSE=0.0764 Training_time=14.4 s 
|2025-03-19 20:31:43| ********************Experiment Results:********************
|2025-03-19 20:31:43| NMAE: 0.0574 ± 0.0000
|2025-03-19 20:31:43| NRMSE: 0.0764 ± 0.0000
|2025-03-19 20:31:43| MAE: 0.0003 ± 0.0000
|2025-03-19 20:31:43| RMSE: 0.0004 ± 0.0000
|2025-03-19 20:31:43| Acc_10: 0.7933 ± 0.0000
|2025-03-19 20:31:43| train_time: 14.3526 ± 0.0000
|2025-03-19 20:31:43| Flops: 76091392
|2025-03-19 20:31:43| Params: 263297
|2025-03-19 20:31:43| Inference time: 1.82 ms
|2025-03-19 20:31:43| ********************Experiment Success********************
```

