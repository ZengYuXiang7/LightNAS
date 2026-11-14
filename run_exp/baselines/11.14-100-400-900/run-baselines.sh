#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."

# 固定设备、训练预算和结果路径；神经网络使用 RTX4090。
BATCH_DIR="run_exp/baselines/11.14-100-400-900"
LATENCY_TABLE="data/nasbench201_complete.pickle"
TRAIN_DEVICE="desktop-cpu-core-i7-7820x"
HARDWARE_TYPE="cpu"
COMPUTE_DEVICE="cuda"
EPOCHS=3000
ROUNDS=5
SEED=0
export DGLBACKEND=pytorch
export PYTHONUNBUFFERED=1

# GCN 全局节点不汇聚架构信息，本批跳过；DNNPerf 尚待核对论文。

# 训练集大小为 100，各方法按顺序运行五轮。
TRAIN_SIZE=100
RESULT_PREFIX="baselines-11.14-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model ml --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gbdt"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# 训练集大小为 400，各方法按顺序运行五轮。
TRAIN_SIZE=400
RESULT_PREFIX="baselines-11.14-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model ml --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gbdt"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"

# 训练集大小为 900，各方法按顺序运行五轮。
TRAIN_SIZE=900
RESULT_PREFIX="baselines-11.14-nasbench201-${TRAIN_DEVICE}-s${TRAIN_SIZE}-seed${SEED}-n${ROUNDS}"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model ml --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --train_size "$TRAIN_SIZE" --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gbdt"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model mlp --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-mlp"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model gru --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-gru"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model lstm --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-lstm"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model birnn --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-birnn"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model flops --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-flops"
.venv/bin/python run.py --batch_dir "$BATCH_DIR" --model brp_nas --latency_table "$LATENCY_TABLE" --train_device "$TRAIN_DEVICE" --dataset "$HARDWARE_TYPE" --device "$COMPUTE_DEVICE" --train_size "$TRAIN_SIZE" --epochs "$EPOCHS" --print_freq 500 --tqdm 0 --rounds "$ROUNDS" --seed "$SEED" --result_name "${RESULT_PREFIX}-brpnas"
