# 单设备延迟预测任务指南

先开展单设备 baseline 实验：每次选择一个设备标签，使用该设备的架构延迟数据独立训练和评估。设备范围按已有延迟标签整理，不依赖 LLGAT 当前的硬件向量。

## 本批固定代表设备

每类固定一个有实测延迟的代表设备。表中设备类别用于描述这一台设备，不表示该类别的跨设备平均。

| 类别 | `--train_device` 标签 |
| --- | --- |
| GPU | `desktop-gpu-gtx-1080ti` |
| CPU | `desktop-cpu-core-i7-7820x` |
| DSP | `mobile-dsp-snapdragon-855-hexagon-690` |
| TPU | `embedded-tpu-edge-tpu` |
| ASIC | `eyeriss` |
| FPGA | `fpga` |

本批分别使用 100、400、900 个训练样本，每个设备、样本数和方法运行 5 轮，种子为 0–4；神经网络最多训练 3000 epoch、早停 patience 为 300，每 500 epoch 输出一次。所有任务在 RTX4090 上串行运行，GBDT 使用 CPU 计算。运行入口为 `run_exp/baselines/11.14-p300-representatives/run-baselines.sh`。

## 可选训练设备

以下 43 个设备标签均有实测延迟。不同 batch 或测量配置分别作为标签列出，不一定代表不同物理设备；各标签的有效架构数量和测量条件见 [设备表](data/DEVICE_TABLE.md)。

| 类别 | `--train_device` 标签 |
| --- | --- |
| GPU | `1080ti_1`、`1080ti_32`、`1080ti_256`、`2080ti_1`、`2080ti_32`、`2080ti_256`、`titan_rtx_1`、`titan_rtx_32`、`titan_rtx_256`、`titanx_1`、`titanx_32`、`titanx_256`、`titanxp_1`、`titanxp_32`、`titanxp_256`、`desktop-gpu-gtx-1080ti`、`desktop-gpu-gtx-1080ti-large`、`embedded-gpu-jetson-nano`、`mobile-gpu-snapdragon-450-adreno-506`、`mobile-gpu-snapdragon-675-adreno-612`、`mobile-gpu-snapdragon-855-adreno-640` |
| CPU | `desktop-cpu-core-i7-7820x`、`desktop-cpu-core-i7-12700h`、`desktop-cpu-core-i9-13900k`、`gold_6226`、`gold_6240`、`silver_4114`、`silver_4210r`、`essential_ph_1`、`pixel2`、`pixel3`、`raspi4`、`samsung_a50`、`samsung_s7`、`mobile-cpu-snapdragon-450-cortex-a53`、`mobile-cpu-snapdragon-675-kryo-460`、`mobile-cpu-snapdragon-855-kryo-485` |
| DSP | `mobile-dsp-snapdragon-675-hexagon-685`、`mobile-dsp-snapdragon-855-hexagon-690` |
| TPU | `embedded-tpu-edge-tpu`、`embedded-tpu-edge-tpu-large` |
| ASIC | `eyeriss` |
| FPGA | `fpga` |

`pixel3` 和 `raspi4` 存在疑似异常的近零延迟，使用前需结合标签分布判断。

## Baselines

本批包含 FLOPs、BRP-NAS、MLP、LSTM、GRU、BiRNN 和 GBDT。FLOPs 基线使用架构操作编号和 FLOPs；参数量已存入数据表，尚未作为模型输入。GCN 当前全局节点不汇聚架构信息，本批跳过。

DNNPerf 代码保留，其实现尚未核对是否与论文一致，暂不列入批量运行。LLGAT 后续再确定版本和设备配置，本批不运行。

## 数据与评估

- 数据表：`data/nasbench201_complete.pickle`，包含 15,625 个架构、各设备延迟、FLOPs 和参数量。
- 每轮随机划分训练集和验证集；测试覆盖该设备的全部有效架构，包括训练集、验证集和剩余架构。测试指标描述全架构表现。
- 测试指标按 KT、MAPE、Acc_10、NMAE、NRMSE、Spearman 的顺序记录；MAPE 和 Acc_10 均为比例。
- 神经网络入口默认训练 500 个样本、最多 3000 epoch、5 轮；早停 patience 默认 100。本批脚本显式指定上面的样本数、输出频率和早停 patience。
- ML 默认使用 100 个验证样本；神经网络验证集为全部有效样本的 10%。比较结果时需注意验证集不同。

## 运行条件

`--train_device` 选择延迟标签，`--device` 选择实际训练使用的计算设备。

计算在 RTX4090 上串行进行。设备清单只表示有延迟数据，模型可运行性仍需通过实际实验验证。
