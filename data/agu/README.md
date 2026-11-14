# 硬件描述

LLGAT 从本目录的 `devices.pickle` 按 `--device_name` 读取数值列表。下列参数是型号规格，不代表测量时实际的频率、内存配置或功耗；延迟标签另存于 `data/`。

## CPU：10 维

向量顺序为：最高标称加速频率（GHz，包含 Turbo Boost Max 3.0 等条件加速）、总核心数、总线程数、L3 缓存（MB）、型号支持的最大理论内存带宽（GB/s）、普通核或性能核数、能效核数、L2 缓存总量（MB）、最大内存通道数、标称基准功耗或 TDP（W）。同构 CPU 的普通核数等于总核心数，能效核数为 0。规格中的最高频率、带宽和基准功耗不代表测量时的实际运行状态。

| `--device_name` | 向量 |
| --- | --- |
| `core-i7-7820x` | `[4.5, 8, 16, 11, 85, 8, 0, 8, 4, 140]` |
| `core-i9-13900k` | `[5.8, 24, 32, 36, 89.6, 8, 16, 32, 2, 125]` |
| `i7-12700h` | `[4.7, 14, 20, 24, 83.2, 6, 8, 11.5, 2, 45]` |
| `gold_6226` | `[3.7, 12, 24, 19.25, 140.8, 12, 0, 12, 6, 125]` |
| `gold_6240` | `[3.9, 18, 36, 24.75, 140.8, 18, 0, 18, 6, 150]` |
| `silver_4114` | `[3.0, 10, 20, 13.75, 115.2, 10, 0, 10, 6, 85]` |
| `silver_4210r` | `[3.2, 10, 20, 13.75, 115.2, 10, 0, 10, 6, 100]` |

CPU 规格参照 [Intel i7-7820X](https://www.intel.com/content/www/us/en/products/sku/123767/intel-core-i77820x-xseries-processor-11m-cache-up-to-4-30-ghz/specifications.html)、[i9-13900K](https://www.intel.com/content/www/us/en/products/sku/230496/intel-core-i913900k-processor-36m-cache-up-to-5-80-ghz/specifications.html)、[i7-12700H](https://www.intel.com/content/www/us/en/products/sku/132228/intel-core-i712700h-processor-24m-cache-up-to-4-70-ghz/specifications.html)、[Xeon Gold 6226](https://www.intel.com/content/www/us/en/products/sku/193957/intel-xeon-gold-6226-processor-19-25m-cache-2-70-ghz/specifications.html)、[Gold 6240](https://www.intel.com/content/www/us/en/products/sku/192443/intel-xeon-gold-6240-processor-24-75m-cache-2-60-ghz/specifications.html)、[Silver 4114](https://www.intel.com/content/www/us/en/products/sku/123550/intel-xeon-silver-4114-processor-13-75m-cache-2-20-ghz/specifications.html) 和 [Silver 4210R](https://www.intel.com/content/www/us/en/products/sku/197098/intel-xeon-silver-4210r-processor-13-75m-cache-2-40-ghz/specifications.html)。

- i7-7820X 的 4.5 GHz 为 Turbo Boost Max 3.0 的标称频率，普通 Turbo Boost 2.0 为 4.3 GHz；8 MB L2 可参照[该处理器的实验配置](https://dafx.de/paper-archive/2019/DAFx2019_paper_40.pdf)。
- i7-12700H 的 11.5 MB L2 可参照 [Intel 社区中的实机信息](https://community.intel.com/t5/Wireless/iwlwifi-WRT-Invalid-buffer-destination-on-6-10-0-rc6/m-p/1614018/highlight/true?profile.language=fr)。Intel 列出的最高内存规格含 LPDDR5-5200，按 128 位计算为 83.2 GB/s；此前的 76.8 GB/s 对应双通道 DDR5-4800。原测量机器的实际内存配置尚未核实。计算采用[英特尔第 12 代内存配置文档](https://edc.intel.com/content/www/us/en/design/ipla/software-development-platforms/client/platforms/alder-lake-desktop/12th-generation-intel-core-processors-datasheet-volume-1-of-2/supported-memory-modules-and-devices/)及[平台规格](https://www.intel.com/content/www/us/en/products/platforms/details/alder-lake-p.html)。
- Xeon L2 总量按 [Intel 说明的每核 1 MB](https://www.intel.com/content/www/us/en/support/articles/000027820/processors/intel-xeon-processors.html)乘以核心数计算。内存带宽按六条 64 位通道和型号支持的最高 DDR4 速率计算，DDR4-2933 得到约 140.8 GB/s，DDR4-2400 得到 115.2 GB/s；这些是理论峰值，并非原服务器实测值。
- Xeon 数值按单颗处理器记录；原服务器的插槽数量、实际内存条和测试时频率尚未确定。`gold_6226` 是 6226，不是 6226R；`gold_6240` 按标签对应 Intel Gold 6240，HELP 论文的设备附表未单独列出该标签。

## 统一延迟表中的 CPU 设备

下表逐项对应 `data/nasbench201_complete.pickle` 的 16 个 CPU 或移动设备标签。核心数与频率为机型或芯片标称值，不等于推理时可用核心数或实际频率。`--train_device` 选择延迟标签，`--device_name` 选择上表中的向量，两者需匹配；本目录目前仅有上表 7 个 CPU 向量。

| 延迟标签 | 核对到的 CPU / SoC 和规格 | `--device_name` |
| --- | --- | --- |
| `desktop-cpu-core-i7-7820x` | Intel Core i7-7820X，8 核 16 线程 | `core-i7-7820x` |
| `desktop-cpu-core-i7-12700h` | Intel Core i7-12700H，6P+8E，20 线程 | `i7-12700h` |
| `desktop-cpu-core-i9-13900k` | Intel Core i9-13900K，8P+16E，32 线程 | `core-i9-13900k` |
| `gold_6226` | Intel Xeon Gold 6226，12 核 24 线程 | `gold_6226` |
| `gold_6240` | 按标签对应 Intel Xeon Gold 6240，18 核 36 线程 | `gold_6240` |
| `silver_4114` | Intel Xeon Silver 4114，10 核 20 线程 | `silver_4114` |
| `silver_4210r` | Intel Xeon Silver 4210R，10 核 20 线程 | `silver_4210r` |
| `essential_ph_1` | Essential PH-1，Snapdragon 835，4 性能核 + 4 能效核；芯片上限 2.45/1.9 GHz | 未提供 |
| `pixel2` | Pixel 2 XL，Snapdragon 835，8 核；Google 标称 2.35/1.9 GHz | 未提供 |
| `pixel3` | Pixel 3，Snapdragon 845，8 核；Google 标称 2.5/1.6 GHz | 未提供 |
| `raspi4` | Raspberry Pi 4，BCM2711，4 个 Cortex-A72；实测频率未注明 | 未提供 |
| `samsung_a50` | Galaxy A50，Exynos 9610，4 个 A73 + 4 个 A53；2.3/1.7 GHz | 未提供 |
| `samsung_s7` | Galaxy S7，Exynos 8890，4 个 Exynos M1 + 4 个 A53；2.3/1.6 GHz | 未提供 |
| `mobile-cpu-snapdragon-450-cortex-a53` | Snapdragon 450，8 个 Cortex-A53；芯片上限 1.8 GHz | 未提供 |
| `mobile-cpu-snapdragon-675-kryo-460` | Snapdragon 675，2 性能核 + 6 能效核；芯片上限 2.0 GHz | 未提供 |
| `mobile-cpu-snapdragon-855-kryo-485` | Snapdragon 855，1 超大核 + 3 性能核 + 4 能效核；芯片上限 2.84/2.42/1.80 GHz | 未提供 |

来源和测量条件：

- [HELP 论文附表 A.2](https://proceedings.neurips.cc/paper/2021/file/e3251075554389fe91d17a794861d47b-Supplemental.pdf)列出 A50、S7、PH-1、Pixel 2 XL、Pixel 3 和 4 GB Raspberry Pi 4；其中 S7 明确是 Exynos 8890，不能按其他地区的 Snapdragon 820 版本填写。
- [Google 手机规格](https://support.google.com/pixelphone/answer/16043605?hl=en)给出的 Pixel 2 XL 和 Pixel 3 机型频率分别低于 [Snapdragon 835](https://www.qualcomm.com/news/releases/2017/01/qualcomm-snapdragon-835-mobile-platform-power-next-generation-immersive) 和 [Snapdragon 845](https://www.qualcomm.com/news/releases/2017/12/qualcomm-snapdragon-845-mobile-platform-introduces-new-innovative) 芯片上限。[Samsung A50](https://image-us.samsung.com/SamsungUS/samsungbusiness/products/mobile/phones/galaxy-a/galaxy-a50/pdf/HHP-UNLOCKED-GALAXYA50DSHT-AUG19T-Final-8-16-19.pdf) 和 [Samsung S7](https://images.samsung.com/is/content/samsung/p5/sg/news/local/sg-news-local-Media-Release-Samsung-Launches-Galaxy-S7-4G-and-S7-edge-4G.pdf)规格可查对应机型频率。
- [HW-NAS-Bench 论文](https://openreview.net/pdf?id=_0kaDkv3dVf)说明 Pixel 3 的这组延迟使用 TFLite，测量时只使用大核；Raspberry Pi 4 也使用 TFLite。树莓派官方资料曾列出 [1.5 GHz](https://magazine.raspberrypi.com/articles/raspberry-pi-4-vs-raspberry-pi-3b-plus)，现行规格列为 [1.8 GHz](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/)，原论文未注明本次实测频率。
- [BRP-NAS 数据来源](https://github.com/zheng-ningxin/brp-nas)列出三组 Snapdragon CPU 标签和 INT8 测量；其具体手机型号、运行频率与内存配置未明确。芯片规格参照 Qualcomm 的 [450 核心](https://www.qualcomm.com/news/releases/2017/06/qualcomm-snapdragon-450-mobile-platform-bring-14nm-finfet-process-enhanced)、[450 频率](https://www.qualcomm.com/smartphones/products/4-series/snapdragon-450-mobile-platform)、[675](https://www.qualcomm.com/smartphones/products/6-series/snapdragon-675-mobile-platform)、[855](https://www.qualcomm.com/news/releases/2018/12/qualcomm-announces-new-flagship-snapdragon-855-mobile-platform-new-decade)。

移动设备缺少完整、可比的 10 维 CPU 参数，尤其 CPU/SoC 的功耗、缓存或实际内存配置；不能把整机功耗当作 CPU TDP，也不能把未知缓存写成 0。因此目前只记录已核实的规格，不生成这 9 个标签的 LLGAT 硬件向量。

## GPU：4 维

向量顺序为 CUDA 核心数、参考加速频率（MHz）、显存（GB）、显存位宽（bit）。3080 采用 10 GB 版本的参考规格；实际显卡频率可能因厂商与运行状态不同而变化。

| `--device_name` | 向量 | 参考规格 |
| --- | --- | --- |
| `1080Ti` | `[3584, 1582, 11, 352]` | [NVIDIA GTX 1080 Ti](https://www.nvidia.com/en-gb/geforce/news/geforce-garage-how-to-build-our-geforce-esports-rig/) |
| `3080` | `[8704, 1710, 10, 320]` | [NVIDIA RTX 3080](https://www.nvidia.com/en-eu/geforce/graphics-cards/30-series/rtx-3080-3080ti/) |
| `4090` | `[16384, 2520, 24, 384]` | [NVIDIA RTX 4090](https://www.nvidia.com/en-me/geforce/graphics-cards/40-series/rtx-4090/) |

CPU 输入从 5 维扩展到 10 维，旧 CPU LLGAT checkpoint 的硬件编码层尺寸不兼容，需要重新训练。i7-7820X 的频率和 i7-12700H 的理论带宽也已更正，使用更正前的 10 维 checkpoint 时输入语义会变化。GPU 向量维数不变，但 3080 和 4090 的输入数值已更正，使用旧 checkpoint 时预测语义会变化。
