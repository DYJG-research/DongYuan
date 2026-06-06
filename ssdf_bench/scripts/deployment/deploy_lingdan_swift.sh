#!/bin/bash

# ================= 基础配置 =================
export CUDA_VISIBLE_DEVICES=5

MODEL_PATH="${MODEL_PATH:?Error: MODEL_PATH is not set}"
PORT=8008
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/Lingdan-13B-PR_8008_.log"

# ================= 环境准备 =================
mkdir -p ${LOG_DIR}

# ================= 启动 vLLM =================
# nohup vllm serve ${MODEL_PATH} \
#     --port ${PORT} \
#     --max-model-len 20480 \
#     --gpu_memory_utilization 0.5 \
#     --tensor_parallel_size 1 \
#     --served-model test \
#     --disable-custom-all-reduce \
#     --max_num_seqs 2 \
#     --trust-remote-code \
#     > ${LOG_FILE} 2>&1 &

swift deploy \
    --model ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --max_new_tokens 10240 \
    --served_model_name test \
    --infer_backend vllm \
    --gpu_memory_utilization 0.95 \
    --tensor_parallel_size 1 \
    --max_model_len 20480 \
    --streaming true \
    > ${LOG_FILE} 2>&1 &
# ================= 提示信息 =================
echo "vLLM 服务已启动"
echo "端口: ${PORT}"
echo "日志: ${LOG_FILE}"
echo "PID: $!"




export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
swift deploy \
    --model "${MODEL_PATH:?Error: MODEL_PATH is not set}" \
    --port "${PORT:-8012}" \
    --model_type baichuan2 \
    --infer_backend sglang \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 20480 \
    --max_new_tokens 20480 \
    --served_model_name test \
    --streaming true \
    --infer_backend pt


# TCMLLM/Lingdan-13B-PR
# git clone https://hf-mirror.com/TCMLLM/Lingdan-13B-PR

# huggingface-cli download --resume-download TCMLLM/Lingdan-13B-PR --include "pytorch_model-00003-of-00003.bin"  --local-dir gLingdan-13B-PR2

# huggingface-cli download --resume-download gpt2 --local-dir gpt2

#   huggingface-cli download gpt2 \
#   --include "pytorch_model.bin" \
#   --local-dir gpt2 \
#   --resume-download