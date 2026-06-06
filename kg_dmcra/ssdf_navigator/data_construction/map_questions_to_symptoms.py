## 把清洗过的question对应为症状 ##
import pandas as pd

# 读取文件
question_df = pd.read_csv('question_cleaned.csv')
symptom_df = pd.read_csv('symptom_duiying.csv')

# 清洗空格
question_df['result'] = question_df['result'].str.strip()
symptom_df['question'] = symptom_df['question'].str.strip()

# 创建question到symptom的映射字典
question_to_symptom = dict(zip(symptom_df['question'], symptom_df['symptom']))

# 根据映射添加symptom列
question_df['symptom'] = question_df['result'].map(question_to_symptom)

# 调整列顺序
question_df = question_df[['symptom', 'result', 'uu_id']]

# 保存结果
question_df.to_csv('question_cleaned.csv', index=False)

print(f"处理完成，共{len(question_df)}条数据")