#!/bin/bash


# 定义数据集数组
CONFIGS=(
  "OurModelConfig"
)
spliter_ratios=(
    "3:4:93"
)
for config in "${CONFIGS[@]}"
do
    for spliter_ratio in "${spliter_ratios[@]}"
    do
        echo "使用splitter_ratio: $spliter_ratio，配置: $config"
        
        # 调用训练脚本，使用配置和数据集
        python run_train.py --exp_name $config --retrain 1 --logger zyx --transfer False \
        --spliter_ratio "$spliter_ratio" --dataset 201_acc
        
    done
done


# 定义数据集数组
CONFIGS=(
  "NNformerConfig"
)
spliter_ratios=(
    # "1:4:95"
    # "3:4:93"
    "5:4:91"
    "10:4:86"
)
for config in "${CONFIGS[@]}"
do
    for spliter_ratio in "${spliter_ratios[@]}"
    do
        echo "使用splitter_ratio: $spliter_ratio，配置: $config"
        
        # 调用训练脚本，使用配置和数据集
        python run_train.py --exp_name $config --retrain 1 --logger zyx --transfer False \
        --spliter_ratio "$spliter_ratio" --dataset 201_acc
        
    done
done

bash ./script/run_bench101_ours.sh
