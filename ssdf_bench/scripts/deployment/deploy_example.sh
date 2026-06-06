#!/bin/bash
# ============================================================
# SSDF-Core Model Deployment Script (Template)
# ============================================================
# This script demonstrates how to deploy SSDF-Core and other
# models using vLLM for OpenAI-compatible API serving.
#
# Usage:
#   export MODEL_PATH=/path/to/your/model
#   export MODEL_NAME=my-model
#   export PORT=8000
#   bash deploy_example.sh
# ============================================================

# Model and deployment settings (override via environment variables)
MODEL_PATH="${MODEL_PATH:?Error: MODEL_PATH is not set}"
MODEL_NAME="${MODEL_NAME:-ssdf-core}"
PORT="${PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

echo "=========================================="
echo "Deploying model: ${MODEL_NAME}"
echo "Model path: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (tp=${TENSOR_PARALLEL_SIZE})"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
vllm serve ${MODEL_PATH} \
    --port ${PORT} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu_memory_utilization ${GPU_MEMORY_UTIL} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --served_model_name ${MODEL_NAME} \
    --max_num_seqs ${MAX_NUM_SEQS}
