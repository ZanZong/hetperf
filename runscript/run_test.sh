#!/bin/bash

set -x
# if [ "$#" -ne 8 ]
# then
#     echo "usage:" $0 "exp_name model_name t p d gbs mbs nodelist"
#     exit 1
# fi
export CPATH=/usr/include/python3.11:$CPATH
export LD_LIBRARY_PATH=/usr/lib/python3.11:$LD_LIBRARY_PATH

export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)

# code repo path
export REPO_PATH="/home/gmk/hetperf"
# python venv path
export SOURCE_PATH="/home/gmk/hetperf-env"
# parallel config path
export PARALLEL_CONFIG_FILE="$REPO_PATH/configs/test/$3"

export EXP_NAME="hetero-train"
export MODEL_NAME="GPT-$1"
export GLOBAL_BATCH_SIZE=$2

export MICRO_BATCH_SIZE=2
export NODELIST=octave,ja[1-4]
export WORLD_SIZE=8

export NUM_LAYERS=-1
export HIDDEN_SIZE=-1
export NUM_ATTN_HEADS=-1

if [ ${MODEL_NAME} == "GPT-760M" ];then
    export NUM_LAYERS=24
    export HIDDEN_SIZE=1536
    export NUM_ATTN_HEADS=16
fi

if [ ${MODEL_NAME} == "GPT-1.3B" ];then
    export NUM_LAYERS=24
    export HIDDEN_SIZE=2048
    export NUM_ATTN_HEADS=16
fi

# use following 4 models
if [ ${MODEL_NAME} == "GPT-2.1B" ];then
    export NUM_LAYERS=10
    export HIDDEN_SIZE=4096
    export NUM_ATTN_HEADS=32
fi

if [ ${MODEL_NAME} == "GPT-4.7B" ];then
    export NUM_LAYERS=24
    export HIDDEN_SIZE=4096
    export NUM_ATTN_HEADS=32
fi

if [ ${MODEL_NAME} == "GPT-6.2B" ];then
    export NUM_LAYERS=32
    export HIDDEN_SIZE=4096
    export NUM_ATTN_HEADS=32
fi

if [ ${MODEL_NAME} == "GPT-11B" ];then
    export NUM_LAYERS=56
    export HIDDEN_SIZE=4096
    export NUM_ATTN_HEADS=32
fi

LOG_DIR=$REPO_PATH/logs/${EXP_NAME}
mkdir -p $LOG_DIR

LOG_PREFIX=${MODEL_NAME}\_t$TENSOR_PARALLEL_SIZE\_p$PIPELINE_PARALLEL_SIZE\_d$DATA_PARALLEL_SIZE\_gbs$GLOBAL_BATCH_SIZE\_mbs$MICRO_BATCH_SIZE\_$(date -Iseconds)
LOG_NAME=${LOG_PREFIX}.log

export PROFILER_LOG_PATH=${LOG_DIR}/${LOG_PREFIX}.prof

mkdir -p $PROFILER_LOG_PATH

NNODES=$(scontrol show hostnames ${NODELIST} | wc -l)

# ja+octave
srun \
    -A public \
    -p octave \
    -K \
    -N 1 \
    -w octave \
    --job-name=$EXP_NAME \
	--ntasks-per-node=4 \
    --gres=gpu:a100:4 \
    --export=ALL \
    bash $REPO_PATH/pretrain.sh : \
    -A public \
    -p ja \
    -K \
    -N 4 \
    -w ja[1-4] \
    --job-name=$EXP_NAME \
    --ntasks-per-node=1 \
    --gres=gpu:v100:1 \
    --export=ALL \
    bash $REPO_PATH/pretrain.sh

# srun \
#     -A public \
#     -p ja \
#     -K \
#     -N 4 \
#     -w ja[1-4] \
#     --job-name=$EXP_NAME \
#     --ntasks-per-node=1 \
#     --gres=gpu:v100:1 \
#     --export=ALL \
# 	bash pretrain_4_4.sh : \
#     -A public \
#     -p twills \
#     -K \
#     -N 1 \
#     -w twills \
#     --job-name=$EXP_NAME \
#     --ntasks-per-node=4 \
#     --gres=gpu:v100:2,gpu:a10:2 \
#     --export=ALL \
#     bash pretrain_4_4.sh