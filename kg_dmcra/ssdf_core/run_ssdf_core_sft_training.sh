#!/bin/bash
# ============================================================
# SSDF-Core SFT Training Script (Template)
# ============================================================
# Usage:
#   bash run_ssdf_core_sft_training.sh
#
# Override defaults via environment variables:
#   export MODEL=/path/to/base/model
#   export OUTPUT_DIR=/path/to/output
#   export DATA_DIR=/path/to/data
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   export NPROC_PER_NODE=4
#   export MASTER_PORT=29500
#   export SWANLAB_TOKEN=your_token
# ============================================================

# Default paths (override via environment variables)
MODEL=${MODEL:-"/path/to/your/base/model"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/ssdf_core_sft"}
DATA_DIR=${DATA_DIR:-"./data/sft_data"}

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

# Training dataset paths (override individually or via DATA_DIR)
DATASET_COT=${DATASET_COT:-"${DATA_DIR}/cot_data.jsonl"}
DATASET_DIALOGUE=${DATASET_DIALOGUE:-"${DATA_DIR}/multi_round_dialogue.jsonl"}
DATASET_EXAM_ENHANCE=${DATASET_EXAM_ENHANCE:-"${DATA_DIR}/exam_enhance_data_options.jsonl"}
DATASET_CHOICE=${DATASET_CHOICE:-"${DATA_DIR}/sft_choice_data_exam_options.jsonl"}
DATASET_DISTILL=${DATASET_DISTILL:-"${DATA_DIR}/distill_r1_sft_sampled.jsonl"}

echo "=========================================="
echo "SSDF-Core SFT Training"
echo "Model: ${MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "Data: ${DATA_DIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${NPROC_PER_NODE} processes)"
echo "=========================================="

NPROC_PER_NODE=${NPROC_PER_NODE} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
swift sft \
    --model ${MODEL} \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --save_strategy epoch \
    --logging_steps 1 \
    --max_length 10240 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir ${OUTPUT_DIR} \
    ${SWANLAB_ARGS} \
    --deepspeed zero3 \
    --use_liger_kernel \
    --attn_impl flash_attn \
    --include_tokens_per_second true \
    --dataset ${DATASET_COT} ${DATASET_DIALOGUE} ${DATASET_EXAM_ENHANCE} ${DATASET_CHOICE} ${DATASET_DISTILL}
