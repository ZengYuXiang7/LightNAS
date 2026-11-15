#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."

# 逐个模型完成全部采样量和设备；已完成的配置保留并跳过。
BATCH_DIR="run_exp/baselines/11.15-scheduler-min"
LATENCY_TABLE="data/nasbench201_complete.pickle"
COMPUTE_DEVICE="cuda"
EPOCHS=3000
PATIENCE=300
ROUNDS=5
SEED=0
export DGLBACKEND=pytorch
export PYTHONUNBUFFERED=1

# LSTM：按 100、400、900 样本依次执行全部设备。

TRAIN_SIZE=100

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"


TRAIN_SIZE=400

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"


TRAIN_SIZE=900

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"

# FLOPS：按 100、400、900 样本依次执行全部设备。

TRAIN_SIZE=100

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"


TRAIN_SIZE=400

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"


TRAIN_SIZE=900

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# MLP：按 100、400、900 样本依次执行全部设备。

TRAIN_SIZE=100

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"


TRAIN_SIZE=400

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"


TRAIN_SIZE=900

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"

# GRU：按 100、400、900 样本依次执行全部设备。

TRAIN_SIZE=100

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"


TRAIN_SIZE=400

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"


TRAIN_SIZE=900

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"

# BiRNN：按 100、400、900 样本依次执行全部设备。

TRAIN_SIZE=100

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
# 已完成：.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"


TRAIN_SIZE=400

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"


TRAIN_SIZE=900

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# BRP-NAS：按 100、400、900 样本依次执行全部设备。

TRAIN_SIZE=100

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"


TRAIN_SIZE=400

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"


TRAIN_SIZE=900

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 6 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"
