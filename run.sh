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

# export EXP_NAME=$1
# export MODEL_NAME=$2
# export TENSOR_PARALLEL_SIZE=$3
# export PIPELINE_PARALLEL_SIZE=$4
# export DATA_PARALLEL_SIZE=$5
# export GLOBAL_BATCH_SIZE=$6
# export MICRO_BATCH_SIZE=$7
# export COMPRESS=0
# export NODELIST=octave,twills
# export GPUS_PER_NODE=4

# code repo path
export REPO_PATH="/home/gmk/hetperf"
# python venv path
export SOURCE_PATH="/home/gmk/hetperf-env"

export EXP_NAME="hetero-train"
export MODEL_NAME="GPT-1.3B"
export GLOBAL_BATCH_SIZE=64
export MICRO_BATCH_SIZE=8
export NODELIST=octave,ja[1-4]
export WORLD_SIZE=5

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

if [ ${NUM_LAYERS} == -1 ];then
    echo "model name not found."
    exit -1
fi

mkdir -p ./logs
mkdir -p ./logs/${EXP_NAME}

LOG_DIR=$(pwd)/logs/${EXP_NAME}

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
	--ntasks-per-node=1 \
    --gres=gpu:a100:1 \
    --export=ALL \
    bash pretrain.sh : \
    -A public \
    -p ja \
    -K \
    -N 4 \
    -w ja[1-4] \
    --job-name=$EXP_NAME \
    --ntasks-per-node=1 \
    --gres=gpu:v100:1 \
    --export=ALL \
    bash pretrain.sh

# # single job
# srun \
#     -A long \
#     -p long \
#     -K \
#     -N $NNODES \
#     -w $NODELIST \
#     --time 20:00 \
#     --job-name=$EXP_NAME \
# 	--ntasks-per-node=$GPUS_PER_NODE \
#     --gres=gpu:v100:$GPUS_PER_NODE \
#     --export=ALL \
# 	bash pretrain.sh

# # ja+octave+twills
# srun \
#     -A long \
#     -p long \
#     -K \
#     -N 4 \
#     -w ja[1-4] \
#     --job-name=$EXP_NAME \
#     --ntasks-per-node=1 \
#     --gres=gpu:v100:1 \
#     --export=ALL \
#     bash pretrain.sh : \
#     -A long \
#     -p long \
#     -K \
#     -N 1 \
#     -w octave \
#     --job-name=$EXP_NAME \
# 	--ntasks-per-node=4 \
#     --gres=gpu:a100:4 \
#     --export=ALL \
# 	bash pretrain.sh : \
#     -A long \
#     -p long \
#     -K \
#     -N 1 \
#     -w twills \
#     --job-name=$EXP_NAME \
#     --ntasks-per-node=2 \
#     --gres=gpu:a10:2 \
#     --export=ALL \
#     bash pretrain.sh

# # octave+twills
# srun \
#     -A long \
#     -p long \
#     -K \
#     -N 1 \
#     -w octave \
#     --job-name=$EXP_NAME \
#     --ntasks-per-node=4 \
#     --gres=gpu:a100:4 \
#     --export=ALL \
#     bash pretrain.sh : \
#     -A long \
#     -p long \
#     -K \
#     -N 1 \
#     -w twills \
#     --job-name=$EXP_NAME \
#     --ntasks-per-node=2 \
#     --gres=gpu:v100:2 \
#     --export=ALL \
#     bash pretrain.sh