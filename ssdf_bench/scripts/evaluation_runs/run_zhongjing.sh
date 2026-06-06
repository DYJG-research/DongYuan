OPENAI_API_KEY="${OPENAI_API_KEY:-}"
python ../tcm_benchmark.py \
  --model_type api \
  --api_url http://localhost:8006/v1 \
  --model_name test \
  --api_key $OPENAI_API_KEY \
  --config_file ../configs/config_example.json \
  --output_dir ../results/run-zhongjing-nothink \
  --resume \
  --skip_think   # 若模型不支持 CoT，可加此参数跳过 CoT 相关维度

  #需要改api_url output_dir model_name
  # 参与计分的类别: ['中西医辩证分型', '西医药学', '中医药学', '医学伦理', '大模型内容安全']
# 参与计分的类别对应的分数: [np.float64(0.04505555555555556), np.float64(0.11349999999999999), np.float64(0.083388), np.float64(0.17), np.float64(0.47)]