#!/bin/bash
clear


python run_train.py --exp_name TransModelConfig --retrain 1 --logger zyx --transfer False \
  --src_dataset datasets/nasbench201/pkl/embedded-gpu-jetson-nono-fp16.pkl \
  --dst_dataset datasets/nasbench201/pkl/embedded-gpu-jetson-nono-fp16.pkl


# python run_train.py --exp_name TransModelConfig --retrain 1 --logger zyx --transfer True \
#   --src_dataset datasets/pickle/embedded-gpu-jetson-nono-fp16_data.pkl \
#   --dst_dataset datasets/pickle/desktop-gpu-gtx-1080ti-fp32_data.pkl
