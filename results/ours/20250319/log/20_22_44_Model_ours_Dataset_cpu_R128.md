```python
|2025-03-19 20:22:44| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'data_scaler': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.02, 'device': cuda, 'epochs': 500,
     'eval_set': True, 'ffn_method': moe, 'hyper_search': False, 'idx': 0,
     'log': <utils.exp_logger.Logger object at 0x79fb630c44a0>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms,
     'num_layers': 32, 'optim': AdamW, 'path': ./datasets/, 'patience': 50,
     'pred_len': 96, 'rank': 128, 'record': True, 'retrain': True,
     'rounds': 1, 'seed': 0, 'seq_len': 96, 'shuffle': False,
     'train_size': 500, 'try_exp': 1, 'ts_var': 0, 'verbose': 10,
}
|2025-03-19 20:22:44| ********************Experiment Start********************
|2025-03-19 20:24:10| Round=1 BestEpoch=180 MAE=0.0003 RMSE=0.0005 NMAE=0.0699 NRMSE=0.0944 Training_time=24.1 s 
|2025-03-19 20:24:10| ********************Experiment Results:********************
|2025-03-19 20:24:10| NMAE: 0.0699 ± 0.0000
|2025-03-19 20:24:10| NRMSE: 0.0944 ± 0.0000
|2025-03-19 20:24:10| MAE: 0.0003 ± 0.0000
|2025-03-19 20:24:10| RMSE: 0.0005 ± 0.0000
|2025-03-19 20:24:10| Acc_10: 0.7316 ± 0.0000
|2025-03-19 20:24:10| train_time: 24.1245 ± 0.0000
|2025-03-19 20:24:10| Flops: 57069568
|2025-03-19 20:24:10| Params: 197505
|2025-03-19 20:24:10| Inference time: 1.45 ms
|2025-03-19 20:24:11| ********************Experiment Success********************
```

