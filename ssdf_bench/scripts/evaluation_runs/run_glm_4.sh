OPENAI_API_KEY="${OPENAI_API_KEY:-}"
python ../tcm_benchmark.py \
  --model_type api \
  --api_url http://localhost:8007/v1 \
  --model_name test \
  --api_key $OPENAI_API_KEY \
  --config_file ../config_example.json \
  --output_dir ../results/run-glm4-9b \
  --resume \
  # --skip_think   # 若模型不支持 CoT，可加此参数跳过 CoT 相关维度

  #需要改api_url output_dir model_name