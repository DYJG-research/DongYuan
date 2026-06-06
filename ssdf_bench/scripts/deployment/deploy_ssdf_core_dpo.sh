#!/bin/bash

# ================= 基础配置 =================
export CUDA_VISIBLE_DEVICES=7

MODEL_PATH="${MODEL_PATH:?Error: MODEL_PATH is not set}"
PORT=8011
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/vllm_piweibing-14B_8011_dpo_.log"

# ================= 环境准备 =================
mkdir -p ${LOG_DIR}

# ================= 启动 vLLM =================
nohup vllm serve ${MODEL_PATH} \
    --port ${PORT} \
    --max-model-len 20480 \
    --gpu_memory_utilization 0.85 \
    --tensor_parallel_size 1 \
    --served-model test \
    --disable-custom-all-reduce \
    --max_num_seqs 16 \
    --trust-remote-code \
    > ${LOG_FILE} 2>&1 &

# ================= 提示信息 =================
echo "vLLM 服务已启动"
echo "端口: ${PORT}"
echo "日志: ${LOG_FILE}"
echo "PID: $!"
