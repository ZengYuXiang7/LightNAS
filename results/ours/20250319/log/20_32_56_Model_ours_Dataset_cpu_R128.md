```python
|2025-03-19 20:32:56| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'data_scaler': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.02, 'device': cuda, 'epochs': 500,
     'eval_set': True, 'ffn_method': moe, 'hyper_search': False, 'idx': 0,
     'log': <utils.exp_logger.Logger object at 0x7fea4f7cc4a0>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms,
     'num_layers': 32, 'optim': AdamW, 'path': ./datasets/, 'patience': 50,
     'pred_len': 96, 'rank': 128, 'record': True, 'retrain': True,
     'rounds': 1, 'seed': 0, 'seq_len': 96, 'shuffle': True,
     'train_size': 500, 'try_exp': 1, 'ts_var': 0, 'verbose': 10,
}
|2025-03-19 20:32:56| ********************Experiment Start********************
|2025-03-19 20:34:22| Round=1 BestEpoch=160 MAE=0.0002 RMSE=0.0003 NMAE=0.0336 NRMSE=0.0572 Training_time=23.8 s 
|2025-03-19 20:34:22| ********************Experiment Results:********************
|2025-03-19 20:34:22| NMAE: 0.0336 ± 0.0000
|2025-03-19 20:34:22| NRMSE: 0.0572 ± 0.0000
|2025-03-19 20:34:22| MAE: 0.0002 ± 0.0000
|2025-03-19 20:34:22| RMSE: 0.0003 ± 0.0000
|2025-03-19 20:34:22| Acc_10: 0.9381 ± 0.0000
|2025-03-19 20:34:22| train_time: 23.7525 ± 0.0000
|2025-03-19 20:34:22| Flops: 76091392
|2025-03-19 20:34:22| Params: 263297
|2025-03-19 20:34:22| Inference time: 1.77 ms
|2025-03-19 20:34:22| ********************Experiment Success********************
```

