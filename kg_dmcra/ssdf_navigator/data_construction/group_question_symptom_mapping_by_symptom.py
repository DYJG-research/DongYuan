#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 symptom_duiying.csv 文件中 symptom 相同的行聚集到一起
"""

import csv
from collections import defaultdict

def group_by_symptom(input_file, output_file):
    """
    按照 symptom 列对行进行分组，将相同的 symptom 行聚集到一起
    
    Args:
        input_file: 输入 CSV 文件路径
        output_file: 输出 CSV 文件路径
    """
    # 使用字典存储每个 symptom 对应的所有行
    symptom_groups = defaultdict(list)
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 读取表头
        
        # 将所有行按 symptom 分组
        for row in reader:
            if len(row) >= 2:
                symptom = row[0].strip()
                symptom_groups[symptom].append(row)
    
    # 按照 symptom 的字典序排序（也可以保持第一次出现的顺序）
    sorted_symptoms = sorted(symptom_groups.keys())
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # 写入表头
        
        # 按 symptom 顺序写入所有行
        for symptom in sorted_symptoms:
            for row in symptom_groups[symptom]:
                writer.writerow(row)
    
    print(f"处理完成！")
    print(f"共有 {len(symptom_groups)} 个不同的 symptom")
    print(f"总行数: {sum(len(rows) for rows in symptom_groups.values())}")
    print(f"结果已保存到: {output_file}")


if __name__ == '__main__':
    input_file = '/home/zkti/fxy/medical-ppo_1/data_5/symptom_duiying.csv'
    output_file = '/home/zkti/fxy/medical-ppo_1/data_5/symptom_duiying.csv'
    
    group_by_symptom(input_file, output_file)

