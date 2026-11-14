#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."

# 使用 CPU 延迟标签，并在 RTX4090 上执行神经网络训练。
BATCH_DIR="run_exp/baselines/11.13"
LATENCY_TABLE="data/nasbench201_complete.pickle"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
HARDWARE_TYPE="cpu"
HARDWARE_NAME="core-i7-7820x"
COMPUTE_DEVICE="cuda"
TRAIN_SIZE=500
EPOCHS=3000
ROUNDS=5
SEED=0
RESULT_PREFIX="baselines-11.13-nasbench201-8nodes-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
export DGLBACKEND=pytorch

# 按顺序执行，结果名包含设备和架构编码。
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"

.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model ours --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device_name "$HARDWARE_NAME" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-llgat"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gcn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gcn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model ml --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-ml"
