## 清洗一遍question，把找不到对应的question删掉 ##
import pandas as pd

# 读取文件
question_df = pd.read_csv('question.csv')
symptom_df = pd.read_csv('symptom_duiying.csv')

# 从symptom_df中提取所有question，注意列名可能有空格
symptom_questions = symptom_df['question'].str.strip() if 'question' in symptom_df.columns else pd.Series([])

# 过滤question_df，只保留在symptom_questions中存在的问句
question_cleaned = question_df[question_df['result'].str.strip().isin(symptom_questions)]

# 保存清洗后的数据
question_cleaned.to_csv('question_cleaned.csv', index=False)

print(f"原始数据: {len(question_df)} 条")
print(f"清洗后: {len(question_cleaned)} 条")