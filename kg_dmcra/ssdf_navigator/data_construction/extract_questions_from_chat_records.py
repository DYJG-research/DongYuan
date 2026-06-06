## 提取type=1的系统问题序列 ##
import pandas as pd

# 读取原始数据
df = pd.read_csv('chat.csv')

# 提取type=1的系统提问
questions_df = df[df['type'] == 1].copy()

# 保持原有的列顺序
questions_df = questions_df[['result', 'type', 'uu_id']]

# 保存到新的CSV文件
questions_df.to_csv('question.csv', index=False, encoding='utf-8')

print(f"已提取 {len(questions_df)} 条系统提问，保存为 question.csv")
print("数据预览：")
print(questions_df.head())