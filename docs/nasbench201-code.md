# NAS-Bench-201 独立模型源码

## 成品

`data/nasbench201_code/` 是不进入 Git 的代码输入数据集：

```text
nasbench201_code/
├── README.md
├── manifest.jsonl
└── models/
    ├── model_00000.py
    ├── model_00001.py
    └── ... model_15624.py
```

每个 `.py` 文件独立包含：

- 已固定的六条边操作与完整 cell 前向计算；不在运行时解析 genotype。
- 完整网络的 stem、三个 cell 阶段、两个降采样块、池化和分类头。
- 所需算子定义及 MIT 版权声明；仅依赖 PyTorch，不依赖 LLGAT 或官方库。
- 无构造参数的 `NASBench201` 类，前向返回 `(pooled_features, logits)`。

这些是模型**源码输入**，不是训练权重，也不是描述文本。每个文件可直接作为
LLM 的一个代码样本；LLM 生成的描述再与同一架构的图信息结合。

`manifest.jsonl` 每行记录 `arch_index`、`arch_str`、`code_file`、`graph_op_ids`
及网络配置。`graph_op_ids` 来自现有统一表的八节点操作槽位，供后续文本与图
分支按架构字符串对齐。代码与索引均不包含准确率或时延标签。

## 导出

在 RTX4090 的 `/home/rtx4090/code/python/current/LLGAT` 中执行：

```bash
.venv/bin/python -m utils.nasbench201_code --tqdm 0
```

默认读取现有 `data/nasbench201_complete.pickle`，导出全部 15,625 个架构。
默认配置为官方 CIFAR-10 参考网络：C=16、每阶段 N=5 个 cell、10 个类别；
对应常规输入为 `[batch, 3, 32, 32]`。命令支持显式设置 `--channels`、
`--cells_per_stage` 和 `--num_classes`，不改变已有训练入口或数据。

输出目录必须尚不存在；创建目录时直接报错，避免覆盖已经生成的输入。
重新生成时请用 `--output_dir` 指定新的项目内目录。

## 编号与实现来源

原数据表的字典顺序不是官方架构编号。导出器依次枚举官方五种操作的六边组合，
再用独立的 `random.Random(88)` 打乱，复现官方 `traverse_net` 的顺序。
现有标签仍按 `arch_str` 关联，不按表的插入顺序推断架构编号。

参考仓库：<https://github.com/D-X-Y/AutoDL-Projects>

固定源码版本：`f46486e21b71ae6459a700be720d7648b5429569`。

相关文件：

- `xautodl/models/cell_searchs/genotypes.py`
- `xautodl/models/cell_operations.py`
- `xautodl/models/cell_infers/cells.py`
- `xautodl/models/cell_infers/tiny_network.py`
- `exps/NATS-Bench/tss-collect.py` 的 `traverse_net`

导出保留 ReLU–Conv–BN、平均池化的 padding 计数规则、恒等连接和乘零操作。
cell 的逐项加法也保留官方 `sum(generator)` 从整数 0 起步的顺序；不根据抽象
图自行删除 `none`、合并分支或声称图分支一定并行执行。

## 验证

`tests/test_nasbench201_code.py` 检查：

1. 15,625 个架构唯一且与现有表完全对应；编号 0、9、123 与官方锚点一致。
2. 全部源文件可解析、编译，仅导入 PyTorch，且架构常量与索引一致。
3. 五种纯操作及若干混合架构，在相同参数和 BN 缓冲区下，与官方完整网络的
   pooled features、logits 逐元素完全相同。
4. 上述模型的参数量与现有数据表一致。

数值验证需要上述固定版本的四份模型源文件，按原路径放在一个参考目录中。
该目录只需保留 `xautodl/models/...` 包结构；无需安装完整 AutoDL 项目。
在 RTX4090 项目根目录设置 `NASBENCH201_REFERENCE_ROOT` 后执行：

```bash
NASBENCH201_REFERENCE_ROOT=/path/to/reference .venv/bin/python -m unittest discover -s tests -p test_nasbench201_code.py -v
```

测试使用 CPU、单线程、随机输入和相同参数，不运行训练、搜索或时延评估实验。

## 边界

- 模型权重是新初始化的，未下载 benchmark 的训练权重。
- 这些是官方 PyTorch **参考实现的展开代码**，不是原始硬件平台编译后程序。
  将其用于解释时延前，仍须确认时延标签的原始测量框架、输入和优化流程。
- 源码和图来自同一个架构，不意味着获得了新的运行时观测；是否有预测增益
  由后续受控实验判断。本次导出不调用 LLM，也不改动现有预测模型或训练协议。
