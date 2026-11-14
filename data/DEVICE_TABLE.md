# NAS-Bench-201 设备表（可选设备清单）

- 数据文件：`data/nasbench201_complete.pickle`
- 延迟单位：毫秒（ms）
- 用途：每次用 `--train_device` 选择一个设备标签，对该设备的架构延迟建模。
- CPU 型号、规格来源及 LLGAT 可用的硬件描述参见 [agu/README.md](agu/README.md)。

## 怎么用

```bash
.venv/bin/python run.py --model ours --train_device desktop-cpu-core-i7-7820x --device_name core-i7-7820x
```

- `--train_device` 必须使用本表的设备标签；默认是 `desktop-cpu-core-i7-7820x`。
- LLGAT 的 `--device_name` 指向独立硬件向量，需与所选延迟标签对应；其他模型不需要该参数。

## 一、43 个设备标签总表

类别按设备名称人工归类；`来源` 列 H = HELP（27 台，覆盖全部 15625 个架构），B = BRP-NAS（14 台，覆盖 14802–15284），S = 自测（2 台，各覆盖 15284）。

### 服务器 GPU（15 台，HELP）

| 设备名 | 来源 | 覆盖架构数 | min (ms) | 中位 (ms) | max (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `1080ti_1` | H | 15625 | 3.60 | 9.53 | 17.09 |
| `1080ti_32` | H | 15625 | 3.54 | 9.05 | 19.56 |
| `1080ti_256` | H | 15625 | 8.00 | 22.56 | 35.58 |
| `2080ti_1` | H | 15625 | 2.68 | 7.55 | 15.85 |
| `2080ti_32` | H | 15625 | 2.89 | 7.48 | 13.86 |
| `2080ti_256` | H | 15625 | 5.96 | 17.65 | 44.31 |
| `titan_rtx_1` | H | 15625 | 2.49 | 6.38 | 12.18 |
| `titan_rtx_32` | H | 15625 | 2.40 | 6.25 | 10.62 |
| `titan_rtx_256` | H | 15625 | 5.94 | 18.06 | 47.72 |
| `titanx_1` | H | 15625 | 3.26 | 8.83 | 16.66 |
| `titanx_32` | H | 15625 | 3.19 | 8.69 | 16.24 |
| `titanx_256` | H | 15625 | 8.15 | 24.58 | 39.96 |
| `titanxp_1` | H | 15625 | 3.60 | 9.41 | 18.38 |
| `titanxp_32` | H | 15625 | 3.29 | 8.68 | 17.94 |
| `titanxp_256` | H | 15625 | 6.82 | 20.58 | 33.81 |

### 桌面 / 边缘 GPU（3 台，BRP-NAS）

| 设备名 | 来源 | 覆盖架构数 | min (ms) | 中位 (ms) | max (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `desktop-gpu-gtx-1080ti` | B | 15284 | 1.01 | 4.79 | 9.49 |
| `desktop-gpu-gtx-1080ti-large` | B | 15284 | 1.75 | 5.75 | 11.56 |
| `embedded-gpu-jetson-nano` | B | 15284 | 3.91 | 14.46 | 34.01 |

### 服务器 CPU（4 台，HELP）

| 设备名 | 来源 | 覆盖架构数 | min (ms) | 中位 (ms) | max (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `gold_6226` | H | 15625 | 2.66 | 9.61 | 20.37 |
| `gold_6240` | H | 15625 | 2.99 | 10.04 | 25.28 |
| `silver_4114` | H | 15625 | 4.55 | 18.52 | 48.08 |
| `silver_4210r` | H | 15625 | 3.88 | 13.59 | 33.83 |

### 桌面 CPU（3 台：BRP-NAS 1 + 自测 2）

| 设备名 | 来源 | 覆盖架构数 | min (ms) | 中位 (ms) | max (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `desktop-cpu-core-i7-7820x` | B | 15284 | 1.02 | 4.78 | 9.51 |
| `desktop-cpu-core-i7-12700h` | S | 15284 | 0.40 | 1.45 | 5.25 |
| `desktop-cpu-core-i9-13900k` | S | 15284 | 0.21 | 0.95 | 2.39 |

### 移动 / 边缘 CPU（9 台：HELP 6 + BRP-NAS 3）

| 设备名 | 来源 | 覆盖架构数 | min (ms) | 中位 (ms) | max (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `essential_ph_1` | H | 15625 | 3.00 | 14.00 | 45.00 |
| `pixel2` | H | 15625 | 4.00 | 15.00 | 51.00 |
| `pixel3` | H | 15625 | 0.003 | 10.68 | 36.77 |
| `raspi4` | H | 15625 | 0.01 | 25.68 | 98.89 |
| `samsung_a50` | H | 15625 | 5.00 | 21.00 | 183.00 |
| `samsung_s7` | H | 15625 | 5.00 | 25.00 | 90.00 |
| `mobile-cpu-snapdragon-450-cortex-a53` | B | 15252 | 4.49 | 21.77 | 245.83 |
| `mobile-cpu-snapdragon-675-kryo-460` | B | 15261 | 4.25 | 24.89 | 79.04 |
| `mobile-cpu-snapdragon-855-kryo-485` | B | 15284 | 2.81 | 9.72 | 51.21 |

### 移动 GPU（3 台，BRP-NAS）

| 设备名 | 来源 | 覆盖架构数 | min (ms) | 中位 (ms) | max (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `mobile-gpu-snapdragon-450-adreno-506` | B | 15252 | 3.45 | 11.93 | 42.82 |
| `mobile-gpu-snapdragon-675-adreno-612` | B | 15261 | 1.56 | 5.80 | 20.58 |
| `mobile-gpu-snapdragon-855-adreno-640` | B | 15284 | 1.18 | 3.78 | 10.01 |

### 移动 DSP（2 台，BRP-NAS）

| 设备名 | 来源 | 覆盖架构数 | min (ms) | 中位 (ms) | max (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `mobile-dsp-snapdragon-675-hexagon-685` | B | 15261 | 1.01 | 2.92 | 5.43 |
| `mobile-dsp-snapdragon-855-hexagon-690` | B | 15284 | 1.17 | 2.70 | 4.91 |

### 专用加速器（4 台）

| 设备名 | 类别 | 来源 | 覆盖架构数 | min (ms) | 中位 (ms) | max (ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `embedded-tpu-edge-tpu` | 边缘 TPU | B | 15284 | 0.52 | 1.12 | 2.49 |
| `embedded-tpu-edge-tpu-large` | 边缘 TPU | B | 14802 | 1.83 | 17.87 | 258.05 |
| `eyeriss` | ASIC | H | 15625 | 1.68 | 5.12 | 10.53 |
| `fpga` | FPGA | H | 15625 | 1.13 | 3.73 | 8.68 |

## 二、类别汇总

| 类别 | 台数 | 设备 |
| --- | ---: | --- |
| 服务器 GPU | 15 | `1080ti_*`、`2080ti_*`、`titan_rtx_*`、`titanx_*`、`titanxp_*`（各 1/32/256 三档 batch） |
| 桌面/边缘 GPU | 3 | `desktop-gpu-gtx-1080ti`、`desktop-gpu-gtx-1080ti-large`、`embedded-gpu-jetson-nano` |
| 服务器 CPU | 4 | `gold_6226`、`gold_6240`、`silver_4114`、`silver_4210r` |
| 桌面 CPU | 3 | `desktop-cpu-core-i7-7820x`、`desktop-cpu-core-i7-12700h`、`desktop-cpu-core-i9-13900k` |
| 移动/边缘 CPU | 9 | `essential_ph_1`、`pixel2`、`pixel3`、`raspi4`、`samsung_a50`、`samsung_s7`、`mobile-cpu-snapdragon-*` |
| 移动 GPU | 3 | `mobile-gpu-snapdragon-450/675/855-*` |
| 移动 DSP | 2 | `mobile-dsp-snapdragon-675/855-*` |
| 边缘 TPU | 2 | `embedded-tpu-edge-tpu`、`embedded-tpu-edge-tpu-large` |
| ASIC | 1 | `eyeriss` |
| FPGA | 1 | `fpga` |
| **合计** | **43** | HELP 27 + BRP-NAS 14 + 自测 2 |

## 三、选设备前要注意的点

1. **同名不同批次不是不同设备**：`1080ti_1` / `1080ti_32` / `1080ti_256` 是同一张 1080Ti 在 batch size 1/32/256 下的测量，`2080ti_*`、`titan_rtx_*`、`titanx_*`、`titanxp_*` 同理；各标签对应不同测量条件。
2. **设备的架构覆盖数不同**：HELP 来源的 27 个标签覆盖全部 15625 个架构；BRP-NAS 来源的 14 个标签覆盖 14802–15284 个；两个自测标签各覆盖 15284 个。
3. **`pixel3` 和 `raspi4` 有异常近零值**：`pixel3` 有 202 个标签 < 0.1 ms（最小 0.0033），`raspi4` 有 157 个（最小 0.0113），而两者中位数分别是 10.68 ms 和 25.68 ms。这些点疑似原始测量异常，使用时建议先检查标签分布。
4. **`embedded-tpu-edge-tpu-large` 延迟方差很大**（1.83–258.05 ms），且覆盖数最少（14802）。
