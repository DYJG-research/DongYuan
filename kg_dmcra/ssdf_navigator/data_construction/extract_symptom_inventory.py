## 提取症状 ##
import pandas as pd

# 读取文件
df = pd.read_csv('question_cleaned.csv')

# 提取不同的症状并去重排序
unique_symptoms = df['symptom'].drop_duplicates().sort_values().reset_index(drop=True)

# 保存到新文件
unique_symptoms.to_csv('symptoms.csv', index=True, header=['symptom'])

print(f"已提取 {len(unique_symptoms)} 种不同的症状")