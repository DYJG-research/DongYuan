OPENAI_API_KEY="${OPENAI_API_KEY:?Error: OPENAI_API_KEY is not set}"
python ../tcm_benchmark.py \
  --model_type api \
  --api_url https://xiaoai.plus/v1 \
  --model_name gemini-3.1-pro-preview \
  --api_key $OPENAI_API_KEY \
  --config_file ../configs/config_example.json \
  --output_dir ../results/run-gemini-3.1-pro-preview-one1\
  --resume \
  # --skip_think   # 若模型不支持 CoT，可加此参数跳过 CoT 相关维度
  #需要改api_url output_dir model_namebash 
