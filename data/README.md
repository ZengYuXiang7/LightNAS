# 延迟数据

`nasbench201_complete.pickle` 包含 NAS-Bench-201 的架构图、操作编号和各设备的实测延迟（毫秒），延迟数据来自 fewshot 多设备实测表。为兼容本项目的 NumPy 版本，数组已转为 Python 列表，数值与设备标签保持不变。每次实验选择一个设备；可用设备标签和覆盖情况参见 `DEVICE_TABLE.md`。

LLGAT 默认读取该表，并通过 `--train_device` 选择设备；读取时从架构字符串转换操作编号，将毫秒转换为原项目使用的秒。缺少该设备实测标签的架构不参与训练。数据层使用官方八节点：输入、六个操作槽位和输出；`none` 与 `skip_connect` 均保留。LLGAT、GCN、BRP-NAS 在各自前向过程中添加全局节点。序列模型同样接收八个节点的操作编号。

每轮保留原有训练集和验证集划分，测试集使用所选设备的全部有效架构，包含训练集、验证集和其余架构；KT、MAPE 等测试指标因此反映全架构表现，不是未见样本上的独立测试结果。

该表每个架构记录包含 `flops` 和 `params`，分别以 MFLOPs 和百万参数为单位。两项均来自 NAS-Bench-201 官方本体的 `cifar10-valid`、12 epoch 数据，按官方 API 对多个种子取均值。FLOPs 基线从同一表按架构字符串读取 `flops`。

串行运行各模型的脚本为 `run_exp/baselines/11.13/run-all-models.sh`，其中 ML 分支运行 GBDT。脚本头部可修改设备标签、硬件描述、训练样本数和轮次；默认使用原项目的 i7-7820x CPU 延迟标签及 CPU 计算。LLGAT 的硬件描述须与所选设备匹配。

单独运行示例：

```bash
.venv/bin/python run.py --model ours --train_device desktop-cpu-core-i7-7820x
```

使用 `--model` 选择模型；例如 `.venv/bin/python run.py --model mlp --rank 128 --bs 32`。通过 `.venv/bin/python run.py --model ours --help` 查看该模型可用的参数。未指定 `--bs` 时，使用原模型配置的批大小。

机器学习实验通过 `.venv/bin/python run.py --model ml` 运行 GBDT，按原候选范围用验证集 MSE 选参数。`--train_size` 和 `--ml_valid_size` 分别控制训练集及验证集数量。ML 沿用原来默认 100 条验证数据，神经网络当前使用全部样本的 10% 作验证，比较结果时应注意这一差异。

实验的 `--batch_dir` 默认是 `run_exp/manual`。批次报告和 JSON 分别写入 `reports/{dataset_name}/` 和 `records/{dataset_name}/`，每轮权重写入 `checkpoints/{dataset_name}/{result_name}/round_{round}.pt`，批次汇总追加到 `run.log`。GBDT 以 pickle 保存回归器，神经网络权重使用 PyTorch。

BRP-NAS 和 DNNPerf 的 `--rank` 默认分别为 600 和 32。批大小默认沿用旧配置：BRP-NAS 为 1，DNNPerf 为 64，其余模型为 32；命令行可分别覆盖。

自测的 `desktop-cpu-core-i9-13900k` 和 `desktop-cpu-core-i7-12700h` 已加入统一表，各有 15284 个架构标签。两份单设备平均延迟表来自 LLGAT Git 提交 `58bbd4e` 的历史版本。i9 的原始逐次测量文件保留在 RTX4090 的 `~/code/python/归档任务/brp-nas-main/tools/i9-13900k.pickle`；i7 原始测量明细尚未找到。

使用 LLGAT 模型时，这两台设备分别设置 `--device_name core-i9-13900k` 和 `--device_name i7-12700h`。四个 Xeon 服务器 CPU 也已有硬件向量，`--device_name` 与 `--train_device` 均使用设备标签。CPU 硬件向量已扩展到 10 维，GPU 频率统一采用 MHz 加速频率；全部 CPU 型号、字段、规格来源及兼容性参见[硬件描述](agu/README.md)。

LLGAT 模型还需要 `data/agu/devices.pickle` 中对应设备的硬件描述；多设备延迟表不包含这些描述。新增设备时，需要单独确认硬件描述和 CPU/GPU 模型配置。设备文件或设备名缺失时直接报错，不会自动调用外部 API。

八节点编码及模型内全局节点与旧版剪枝编码不同，旧版模型权重不能直接用于新版输入。批量脚本使用独立结果名，并指定重新训练。
