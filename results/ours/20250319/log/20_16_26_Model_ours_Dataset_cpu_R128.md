```python
|2025-03-19 20:16:26| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'data_scaler': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.02, 'device': cuda, 'epochs': 200,
     'eval_set': True, 'ffn_method': moe, 'hyper_search': False, 'idx': 0,
     'log': <utils.exp_logger.Logger object at 0x77bdce8cc4a0>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'multi_dataset': False, 'norm_method': rms, 'num_layers': 32,
     'optim': AdamW, 'path': ./datasets/, 'patience': 50, 'pred_len': 96,
     'rank': 128, 'record': True, 'retrain': True, 'rounds': 1,
     'seed': 0, 'seq_len': 96, 'shuffle': False, 'train_size': 500,
     'try_exp': 1, 'ts_var': 0, 'verbose': 10,
}
|2025-03-19 20:16:26| ********************Experiment Start********************
|2025-03-19 20:16:50| Round=1 BestEpoch= 27 MAE=0.0007 RMSE=0.0009 NMAE=0.1455 NRMSE=0.1726 Training_time=3.1 s 
|2025-03-19 20:16:50| ********************Experiment Results:********************
|2025-03-19 20:16:50| NMAE: 0.1455 ± 0.0000
|2025-03-19 20:16:50| NRMSE: 0.1726 ± 0.0000
|2025-03-19 20:16:50| MAE: 0.0007 ± 0.0000
|2025-03-19 20:16:50| RMSE: 0.0009 ± 0.0000
|2025-03-19 20:16:50| Acc_10: 0.4033 ± 0.0000
|2025-03-19 20:16:50| train_time: 3.1032 ± 0.0000
|2025-03-19 20:16:50| Flops: 14454784
|2025-03-19 20:16:50| Params: 50177
|2025-03-19 20:16:50| Inference time: 0.32 ms
|2025-03-19 20:16:51| ********************Experiment Success********************
```

