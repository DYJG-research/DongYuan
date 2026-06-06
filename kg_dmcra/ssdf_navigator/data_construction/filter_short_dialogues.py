## 删除uu_id只有一行或两行的数据 ##
import pandas as pd

# 读取文件
df = pd.read_csv('question_cleaned.csv')

# 统计每个uu_id出现的次数
uu_id_counts = df['uu_id'].value_counts()

# 找出出现3次或以上的uu_id
valid_uu_ids = uu_id_counts[uu_id_counts >= 3].index

# 只保留这些uu_id的数据
df_filtered = df[df['uu_id'].isin(valid_uu_ids)]

# 保存清洗后的数据
df_filtered.to_csv('question_cleaned.csv', index=False)

print(f"原始数据: {len(df)} 条")
print(f"清洗后: {len(df_filtered)} 条")
print(f"删除了 {len(df) - len(df_filtered)} 条数据")
print(f"原始uu_id数量: {len(uu_id_counts)}")
print(f"保留的uu_id数量: {len(valid_uu_ids)}")

