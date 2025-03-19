```python
|2025-03-19 20:28:38| {
     'ablation': 0, 'att_method': self, 'bs': 32, 'classification': False,
     'continue_train': False, 'data_scaler': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.02, 'device': cuda, 'epochs': 500,
     'eval_set': True, 'ffn_method': moe, 'hyper_search': False, 'idx': 0,
     'log': <utils.exp_logger.Logger object at 0x77e0320db4d0>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'monitor_metrics': NMAE, 'multi_dataset': False, 'norm_method': rms,
     'num_layers': 32, 'optim': AdamW, 'path': ./datasets/, 'patience': 50,
     'pred_len': 96, 'rank': 128, 'record': True, 'retrain': True,
     'rounds': 1, 'seed': 0, 'seq_len': 96, 'shuffle': False,
     'train_size': 500, 'try_exp': 1, 'ts_var': 0, 'verbose': 10,
}
|2025-03-19 20:28:38| ********************Experiment Start********************
|2025-03-19 20:29:53| Round=1 BestEpoch=133 MAE=0.0003 RMSE=0.0004 NMAE=0.0588 NRMSE=0.0768 Training_time=19.6 s 
|2025-03-19 20:29:53| ********************Experiment Results:********************
|2025-03-19 20:29:53| NMAE: 0.0588 ± 0.0000
|2025-03-19 20:29:53| NRMSE: 0.0768 ± 0.0000
|2025-03-19 20:29:53| MAE: 0.0003 ± 0.0000
|2025-03-19 20:29:53| RMSE: 0.0004 ± 0.0000
|2025-03-19 20:29:53| Acc_10: 0.7765 ± 0.0000
|2025-03-19 20:29:53| train_time: 19.6420 ± 0.0000
|2025-03-19 20:29:54| Flops: 76091392
|2025-03-19 20:29:54| Params: 263297
|2025-03-19 20:29:54| Inference time: 1.74 ms
|2025-03-19 20:29:54| ********************Experiment Success********************
```

