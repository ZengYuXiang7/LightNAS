#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."

# 选择延迟标签与训练预算，神经网络计算使用 RTX4090。
BATCH_DIR="run_exp/baselines/11.14"
LATENCY_TABLE="data/nasbench201_complete.pickle"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
HARDWARE_TYPE="cpu"
COMPUTE_DEVICE="cuda"
TRAIN_SIZE=500
EPOCHS=3000
ROUNDS=5
SEED=0
RESULT_PREFIX="baselines-11.14-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
export DGLBACKEND=pytorch
export PYTHONUNBUFFERED=1

# GBDT 在 CPU 上选参并训练，其余基线依次使用 CUDA。
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model ml --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gbdt"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# GCN 的全局节点不汇聚架构信息，本批跳过。
