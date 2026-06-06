OPENAI_API_KEY="${OPENAI_API_KEY:-}"
python ../tcm_benchmark.py \
  --model_type api \
  --api_url http://localhost:8006/v1 \
  --model_name test \
  --api_key $OPENAI_API_KEY \
  --config_file ../configs/config_example.json \
  --output_dir ../results/run-Sunsimiao-Qwen2-7B-nothink_1 \
  --resume \
  --skip_think   # 若模型不支持 CoT，可加此参数跳过 CoT 相关维度

  #需要改api_url output_dir model_name