OPENAI_API_KEY="${OPENAI_API_KEY:-}"
python tcm_benchmark.py \
  --model_type api \
  --api_url http://localhost:8005/v1 \
  --model_name DaYiJinGui-v2 \
  --api_key $OPENAI_API_KEY \
  --config_file ./config_example.json \
  --output_dir ./results/run-qwen3-32B \
  --resume \
  # --skip_think   # 若模型不支持 CoT，可加此参数跳过 CoT 相关维度