#!/bin/bash
# ============================================================
# SSDF-Core DPO Training Script (Template)
# ============================================================
# Usage:
#   bash run_ssdf_core_dpo_training.sh
#
# Override defaults via environment variables:
#   export MODEL=/path/to/sft/checkpoint
#   export OUTPUT_DIR=/path/to/output
#   export DATA_DIR=/path/to/data
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   export NPROC_PER_NODE=4
#   export MASTER_PORT=29500
#   export SWANLAB_TOKEN=your_token
# ============================================================

# Default paths (override via environment variables)
MODEL=${MODEL:-"/path/to/your/sft/checkpoint"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/ssdf_core_dpo"}
DATA_DIR=${DATA_DIR:-"./data/dpo_data"}

# Distributed training settings
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# SwanLab experiment tracking (optional)
SWANLAB_TOKEN=${SWANLAB_TOKEN:-""}
SWANLAB_ARGS=""
if [ -n "$SWANLAB_TOKEN" ]; then
    SWANLAB_ARGS="--report_to swanlab --swanlab_token ${SWANLAB_TOKEN} --swanlab_project spleen_stomach_disorders"
fi

# DPO dataset path
DATASET_DPO=${DATASET_DPO:-"${DATA_DIR}/dpo_train_data.jsonl"}

echo "=========================================="
echo "SSDF-Core DPO Training"
echo "Base model (SFT checkpoint): ${MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "Data: ${DATASET_DPO}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${NPROC_PER_NODE} processes)"
echo "=========================================="

NPROC_PER_NODE=${NPROC_PER_NODE} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
swift rlhf \
    --rlhf_type dpo \
    --model ${MODEL} \
    --train_type full \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 10240 \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    ${SWANLAB_ARGS} \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --use_liger_kernel \
    --rpo_alpha 0.1 \
    --padding_free true \
    --dataset ${DATASET_DPO}
