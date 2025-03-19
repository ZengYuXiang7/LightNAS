```python
|2025-03-19 20:09:30| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'data_scaler': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.02, 'device': cuda, 'epochs': 200,
     'eval_set': True, 'ffn_method': moe, 'hyper_search': False, 'idx': 0,
     'log': <utils.exp_logger.Logger object at 0x7e217d1ea1b0>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'pred_len': 96,
     'rank': 128, 'record': True, 'retrain': True, 'rounds': 1,
     'seed': 0, 'seq_len': 96, 'shuffle': False, 'train_size': 500,
     'try_exp': 1, 'ts_var': 0, 'verbose': 10,
}
|2025-03-19 20:09:30| ********************Experiment Start********************
|2025-03-19 20:10:25| Round=1 BestEpoch= 96 MAE=0.0003 RMSE=0.0005 NMAE=0.0667 NRMSE=0.0908 Training_time=12.7 s 
|2025-03-19 20:10:25| ********************Experiment Results:********************
|2025-03-19 20:10:25| NMAE: 0.0667 ± 0.0000
|2025-03-19 20:10:25| NRMSE: 0.0908 ± 0.0000
|2025-03-19 20:10:25| MAE: 0.0003 ± 0.0000
|2025-03-19 20:10:25| RMSE: 0.0005 ± 0.0000
|2025-03-19 20:10:25| Acc_10: 0.7405 ± 0.0000
|2025-03-19 20:10:25| train_time: 12.7417 ± 0.0000
|2025-03-19 20:10:25| Flops: 57069568
|2025-03-19 20:10:25| Params: 197505
|2025-03-19 20:10:25| Inference time: 1.43 ms
|2025-03-19 20:10:26| ********************Experiment Success********************
```

