#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."

# 修正验证损失调度方向后重跑受影响模型，GBDT 沿用原结果。
BATCH_DIR="run_exp/baselines/11.15-scheduler-min"
LATENCY_TABLE="data/nasbench201_complete.pickle"
COMPUTE_DEVICE="cuda"
EPOCHS=3000
PATIENCE=300
ROUNDS=5
SEED=0
export DGLBACKEND=pytorch
export PYTHONUNBUFFERED=1

# 先串行执行 FLOPS、MLP 与序列基线。

TRAIN_SIZE=100

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"


TRAIN_SIZE=400

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"


TRAIN_SIZE=900

# CPU：desktop-cpu-core-i7-7820x。
HARDWARE_TYPE="cpu"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# GPU：desktop-gpu-gtx-1080ti。
HARDWARE_TYPE="gpu"
TRAIN_DEVICE="desktop-gpu-gtx-1080ti"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# DSP：mobile-dsp-snapdragon-855-hexagon-690。
HARDWARE_TYPE="dsp"
TRAIN_DEVICE="mobile-dsp-snapdragon-855-hexagon-690"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# TPU：embedded-tpu-edge-tpu。
HARDWARE_TYPE="tpu"
TRAIN_DEVICE="embedded-tpu-edge-tpu"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# ASIC：eyeriss。
HARDWARE_TYPE="asic"
TRAIN_DEVICE="eyeriss"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# FPGA：fpga。
HARDWARE_TYPE="fpga"
TRAIN_DEVICE="fpga"
RESULT_PREFIX="baselines-11.15-p300-min-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --device "$COMPUTE_DEVICE" --num_workers 0 --epochs "$EPOCHS" --patience "$PATIENCE" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

# 快速基线全部结束后，再串行执行 BRP-NAS。

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
