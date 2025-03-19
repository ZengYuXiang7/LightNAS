```python
|2025-03-19 19:59:52| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'data_scaler': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.02, 'device': cuda, 'epochs': 200,
     'eval_set': True, 'ffn_method': moe, 'hyper_search': False, 'idx': 0,
     'log': <utils.exp_logger.Logger object at 0x74180d44c4a0>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'pred_len': 96,
     'rank': 128, 'record': True, 'retrain': True, 'rounds': 1,
     'seed': 0, 'seq_len': 96, 'shuffle': False, 'train_size': 500,
     'try_exp': 1, 'ts_var': 0, 'verbose': 10,
}
|2025-03-19 19:59:52| ********************Experiment Start********************
|2025-03-19 20:00:59| Round=1 BestEpoch=140 MAE=0.0015 RMSE=0.0020 NMAE=0.3177 NRMSE=0.3875 Training_time=17.7 s 
|2025-03-19 20:00:59| ********************Experiment Results:********************
|2025-03-19 20:00:59| NMAE: 0.3177 ± 0.0000
|2025-03-19 20:00:59| NRMSE: 0.3875 ± 0.0000
|2025-03-19 20:00:59| MAE: 0.0015 ± 0.0000
|2025-03-19 20:00:59| RMSE: 0.0020 ± 0.0000
|2025-03-19 20:00:59| Acc_10: 0.3065 ± 0.0000
|2025-03-19 20:00:59| train_time: 17.6840 ± 0.0000
|2025-03-19 20:01:00| Flops: 61788160
|2025-03-19 20:01:00| Params: 345089
|2025-03-19 20:01:00| Inference time: 1.36 ms
|2025-03-19 20:01:00| ********************Experiment Success********************
```

