OPENAI_API_KEY="${OPENAI_API_KEY:?Error: OPENAI_API_KEY is not set}"
python ../tcm_benchmark.py \
  --model_type api \
  --api_url https://ark.cn-beijing.volces.com/api/v3 \
  --model_name doubao-seed-2-0-pro-260215 \
  --api_key $OPENAI_API_KEY \
  --config_file ../configs/config_example.json \
  --output_dir ../results/doubao-seed-2-0-pro-260215\
  --resume \
  # --skip_think   # 若模型不支持 CoT，可加此参数跳过 CoT 相关维度
  #需要改api_url output_dir model_namebash 
