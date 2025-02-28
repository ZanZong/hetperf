#! /bin/bash
set -x

# export NNODES=$(( $DATA_PARALLEL_SIZE * $TENSOR_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE / $GPUS_PER_NODE))
export MASTER_ADDR="octave"
# export NODE_RANK=$(expr $SLURM_PROCID / $NNODES)
# export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export RANK=$SLURM_PROCID
export CUDA_DEVICE_MAX_CONNECTIONS=1 # for async gradient all reduce

current_hostname=$(hostname)
if [ "$current_hostname" == "octave" ] || [ "$current_hostname" == "twills" ]; then
    export GLOO_SOCKET_IFNAME=eno1
    export NCCL_SOCKET_IFNAME=eno1
else
    export GLOO_SOCKET_IFNAME=enp68s0f0
    export NCCL_SOCKET_IFNAME=enp68s0f0
fi
export NCCL_IB_DISABLE=1

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

source ~/workspace/mega-env/bin/activate
cd /home/zanzong/workspace/Megatron-LM

VOCAB_FILE=~/datasets/wikidataset/gpt2-vocab.json
MERGE_FILE=~/datasets/wikidataset/gpt2-merges.txt
DATA_PATH=~/datasets/wikidataset/my-bert_text_sentence

# TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 50))
TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 120))

PROFILER_LOG_PATH=$PROFILER_LOG_PATH \
exec python \
        ./pretrain_gpt.py \
        --vocab-file $VOCAB_FILE \
	--merge-file $MERGE_FILE \
        --transformer-impl local \
	--num-layers $NUM_LAYERS \
	--hidden-size $HIDDEN_SIZE \
        --num-attention-heads $NUM_ATTN_HEADS \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --train-samples $TRAIN_SAMPLES \
	--lr-decay-samples 4882800 \
        --lr 0.0001 \
        --min-lr 0.000001 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --timing-log-level 2 \
        --eval-iters -1 \
        --data-path ${DATA_PATH} \
        --split 100,0,0 \
        --clip-grad 1.0 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.002 \
        --fp16 \
        --recompute-granularity selective \
        --hetero-cluster True \
        --parallel-config /home/zanzong/workspace/Megatron-LM/configs/tangram_config_octave_twills.json
        # --use-distributed-optimizer \
        # --enable-hetero-compression $COMPRESS \
        # --stage-layer-num $LL
        # --recompute-granularity full \
        # --recompute-method block \
        # --stage-recompute-num-layers 1 1 2 2