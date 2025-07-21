#!/bin/bash
clear

CONFIGS=(
  "FlopsConfig"
  "MacConfig"
  "GRUConfig"
  "LSTMConfig"
  "BRPNASConfig"
  "GATConfig"
  "NarFormerConfig"
  "nnMeterConfig"
)

for CONFIG in "${CONFIGS[@]}"
do
  echo "Running experiment with config: $CONFIG"
  
  python run_train.py --exp_name "$CONFIG" --retrain 1 --logger zyx --transfer False \
    --src_dataset datasets/nasbench201/pkl/embedded-gpu-jetson-nono-fp16.pkl \
    --dst_dataset datasets/nasbench201/pkl/embedded-gpu-jetson-nono-fp16.pkl 
  
  echo "Finished $CONFIG"
  echo "-----------------------------"
done