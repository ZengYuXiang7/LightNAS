```python
|2025-03-19 20:20:12| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'data_scaler': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.02, 'device': cuda, 'epochs': 500,
     'eval_set': True, 'ffn_method': moe, 'hyper_search': False, 'idx': 0,
     'log': <utils.exp_logger.Logger object at 0x7680fbe21550>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'pred_len': 96,
     'rank': 128, 'record': True, 'retrain': True, 'rounds': 1,
     'seed': 0, 'seq_len': 96, 'shuffle': False, 'train_size': 500,
     'try_exp': 1, 'ts_var': 0, 'verbose': 10,
}
|2025-03-19 20:20:12| ********************Experiment Start********************
|2025-03-19 20:21:39| Round=1 BestEpoch=180 MAE=0.0003 RMSE=0.0005 NMAE=0.0699 NRMSE=0.0944 Training_time=24.5 s 
|2025-03-19 20:21:39| ********************Experiment Results:********************
|2025-03-19 20:21:39| NMAE: 0.0699 ± 0.0000
|2025-03-19 20:21:39| NRMSE: 0.0944 ± 0.0000
|2025-03-19 20:21:39| MAE: 0.0003 ± 0.0000
|2025-03-19 20:21:39| RMSE: 0.0005 ± 0.0000
|2025-03-19 20:21:39| Acc_10: 0.7316 ± 0.0000
|2025-03-19 20:21:39| train_time: 24.4817 ± 0.0000
|2025-03-19 20:21:39| Flops: 57069568
|2025-03-19 20:21:39| Params: 197505
|2025-03-19 20:21:39| Inference time: 1.38 ms
|2025-03-19 20:21:39| ********************Experiment Success********************
```

