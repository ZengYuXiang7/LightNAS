```python
|2025-03-19 20:27:07| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'data_scaler': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.02, 'device': cuda, 'epochs': 500,
     'eval_set': True, 'ffn_method': moe, 'hyper_search': False, 'idx': 0,
     'log': <utils.exp_logger.Logger object at 0x7953b3689910>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms,
     'num_layers': 32, 'optim': AdamW, 'path': ./datasets/, 'patience': 50,
     'pred_len': 96, 'rank': 128, 'record': True, 'retrain': True,
     'rounds': 1, 'seed': 0, 'seq_len': 96, 'shuffle': False,
     'train_size': 500, 'try_exp': 1, 'ts_var': 0, 'verbose': 10,
}
|2025-03-19 20:27:07| ********************Experiment Start********************
|2025-03-19 20:28:19| Round=1 BestEpoch=126 MAE=0.0003 RMSE=0.0005 NMAE=0.0709 NRMSE=0.0955 Training_time=18.3 s 
|2025-03-19 20:28:19| ********************Experiment Results:********************
|2025-03-19 20:28:19| NMAE: 0.0709 ± 0.0000
|2025-03-19 20:28:19| NRMSE: 0.0955 ± 0.0000
|2025-03-19 20:28:19| MAE: 0.0003 ± 0.0000
|2025-03-19 20:28:19| RMSE: 0.0005 ± 0.0000
|2025-03-19 20:28:19| Acc_10: 0.7611 ± 0.0000
|2025-03-19 20:28:19| train_time: 18.3383 ± 0.0000
|2025-03-19 20:28:19| Flops: 76091392
|2025-03-19 20:28:19| Params: 263297
|2025-03-19 20:28:19| Inference time: 1.78 ms
|2025-03-19 20:28:20| ********************Experiment Success********************
```

