#!/bin/bash

CONFIGS=(
#   "FlopsConfig"
#   "MacConfig"
#   "GRUConfig"
#   "LSTMConfig"
#   "BRPNASConfig"
#   "GATConfig"
  "NarFormerConfig"
#   "nnMeterConfig"
)
# 定义微调数据集数组
src_dataset=(
    # "desktop-cpu-core-i7-7820x-fp32.pkl"
    "desktop-gpu-gtx-1080ti-fp32.pkl"
)

dst_dataset=(
    "desktop-cpu-core-i7-7820x-fp32.pkl"
    # "mobile-dsp-snapdragon-675-hexagon-685-int8.pkl"
#     "mobile-dsp-snapdragon-855-hexagon-690-int8.pkl"
#     "embedded-gpu-jetson-nono-fp16.pkl"
#     "desktop-gpu-gtx-1080ti-fp32.pkl"
#     "mobile-gpu-snapdragon-450-adreno-506-int8.pkl"
#     "mobile-gpu-snapdragon-675-adren0-612-int8.pkl"
#     "mobile-cpu-snapdragon-675-kryo-460-int8.pkl"
#     "mobile-cpu-snapdragon-855-kryo-485-int8.pkl"
#     "embedded-tpu-edgetpu-large.pkl"
#     "mobile-cpu-snapdragon-450-contex-a53-int8.pkl"
#     "embeeded-tpu-edgetpu-int8.pkl"
#     "desktop-gpu-gtx-1080ti-large.pkl"
#     "mobile-gpu-snapdragon-855-adren0-640-int8.pkl"
)
spliter_ratios=(
    # "1:4:95"
    "5:4:91"
    # "10:4:86"
)
# 微调阶段
for CONFIG in "${CONFIGS[@]}"
do
    for spliter_ratio in "${spliter_ratios[@]}"
    do
        for i in "${!src_dataset[@]}"
        do
            for j in "${!dst_dataset[@]}"  # 使用 dst_dataset 的索引来遍历
            do
                dataset1=${src_dataset[$i]}   # 使用 src_dataset 中的数据集
                dataset2=${dst_dataset[$j]}   # 使用 dst_dataset 中的数据集
                
                # 确保 dataset1 和 dataset2 不相同
                if [ "$dataset1" != "$dataset2" ]; then
                    echo "正在微调 $dataset1 和 $dataset2 使用配置 $CONFIG"
                    
                    # 加载第一个训练好的模型，并在第二个数据集上进行微调
                    python run_train.py --exp_name "$CONFIG" --retrain 1 --logger zyx --transfer True \
                    --src_dataset "datasets/nasbench201/pkl/$dataset1" \
                    --dst_dataset "datasets/nasbench201/pkl/$dataset1" \
                    --spliter_ratio "$spliter_ratio" \
                    --pretrained_model "models/$dataset1_model.pth"   # 使用第一个数据集训练的模型进行微调
                
                    # 加载第一个训练好的模型，并在第二个数据集上进行微调
                    python run_transfer.py --exp_name "$CONFIG" --retrain 1 --logger zyx --transfer True \
                    --src_dataset "datasets/nasbench201/pkl/$dataset1" \
                    --dst_dataset "datasets/nasbench201/pkl/$dataset2" \
                    --spliter_ratio "$spliter_ratio" \
                    --pretrained_model "models/$dataset1_model.pth"   # 使用第一个数据集训练的模型进行微调
                    
                    # 输出当前任务完成
                    echo "$dataset1 和 $dataset2 使用配置 $CONFIG 微调完成！"
                fi
            done
        done
    done
done

echo "所有微调实验已完成！"
