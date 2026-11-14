#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."

# macOS 单设备 baseline 烟测：神经网络每个模型训练 1 个 epoch、1 轮。
BATCH_DIR="run_exp/smoke/11.14"
LATENCY_TABLE="data/nasbench201_complete.pickle"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
HARDWARE_TYPE="cpu"
COMPUTE_DEVICE="cpu"
TRAIN_SIZE=500
EPOCHS=1
ROUNDS=1
SEED=0
RESULT_PREFIX="smoke-11.14-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-epoch${EPOCHS}-round${ROUNDS}"
export DGLBACKEND=pytorch

# 按顺序执行，出现错误时中止。
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"

# GBDT 没有 epoch，运行 1 轮并保留原参数搜索。
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model ml --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gbdt"

# GCN 当前全局节点不汇聚架构信息，单独放在最后以暴露其问题。
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gcn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gcn"
