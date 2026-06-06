python train.py \
  --question_file ./question_cleaned_20260223.csv \
  --symptom_file ./symptoms_20260223.csv \
  --fusion_mode mul \
  --max_seq_len 20 \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-4 \
  --save_path transformer_policy_mul_20260223_max_seq_len_20.pth \
  --gpu 0