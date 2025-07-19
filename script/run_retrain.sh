#!/bin/bash

# 定义数据集数组
datasets=(
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

# 遍历每个数据集，并运行训练脚本
for dataset in "${datasets[@]}"
do
    echo "正在训练数据集: $dataset"
    
    # 调用训练脚本
    python run_train.py --exp_name GNNModelConfig --retrain 1 --logger zyx --transfer False \
    --src_dataset "datasets/nasbench201/pkl/$dataset" \
    --dst_dataset "datasets/nasbench201/pkl/$dataset"
    
    # 输出当前任务完成
    echo "$dataset 的训练完成！"
    
    # 你可以在此处添加其他需要在每个数据集运行时执行的命令，例如暂停、日志记录等。
done

echo "所有实验已完成！"
