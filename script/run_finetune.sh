#!/bin/bash

conda activate wangyang

# 定义微调数据集数组
finetune_datasets=(
    "desktop-cpu-core-i7-7820x-fp32_data.pkl"
    "mobile-dsp-snapdragon-675-hexagon-685-int8_data.pkl"
    "mobile-dsp-snapdragon-855-hexagon-690-int8_data.pkl"
    "embedded-gpu-jetson-nono-fp16_data.pkl"
    "desktop-gpu-gtx-1080ti-fp32_data.pkl"
    "mobile-gpu-snapdragon-450-adreno-506-int8_data.pkl"
    "mobile-gpu-snapdragon-675-adren0-612-int8_data.pkl"
    "mobile-cpu-snapdragon-675-kryo-460-int8_data.pkl"
    "mobile-cpu-snapdragon-855-kryo-485-int8_data.pkl"
    "embedded-tpu-edgetpu-large_data.pkl"
    "mobile-cpu-snapdragon-450-contex-a53-int8_data.pkl"
    "embeeded-tpu-edgetpu-int8_data.pkl"
    "desktop-gpu-gtx-1080ti-large_data.pkl"
    "mobile-gpu-snapdragon-855-adren0-640-int8_data.pkl"
)

# 微调阶段
for i in "${!finetune_datasets[@]}"
do
    for j in $(seq $((i+1)) ${#finetune_datasets[@]})
    do
        dataset1=${finetune_datasets[$i]}
        dataset2=${finetune_datasets[$j]}
        
        echo "正在微调 $dataset1 和 $dataset2"
        
        # 加载第一个训练好的模型，并在第二个数据集上进行微调
        python run_train.py --exp_name GNNModelConfig --retrain 1 --logger zyx --transfer True \
        --src_dataset "datasets/pickle/$dataset2" \
        --dst_dataset "datasets/pickle/$dataset2" \
        --pretrained_model "models/$dataset1_model.pth"  # 使用第一个数据集训练的模型进行微调
        
        # 输出当前任务完成
        echo "$dataset1 和 $dataset2 微调完成！"
    done
done

echo "所有微调实验已完成！"
