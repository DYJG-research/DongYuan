import csv
import hashlib

def read_csv_data(filepath, max_row=2078):
    """读取CSV文件的前max_row行数据"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_row - 1:  # -1 因为不包括表头
                break
            data.append(row)
    return data

def identify_conversation_segments(data):
    """识别对话段
    对话段通常以特定文本开头，或者通过uu_id的变化来识别
    """
    segments = []
    current_segment = []
    current_uu_id = None
    
    for row in data:
        result = row['result']
        uu_id = row['uu_id']
        
        # 判断是否是新的对话段开始
        # 1. 如果result包含"**欢迎进行中医体质检测!**"或"下面我们开始进行问诊"
        # 2. 或者uu_id发生变化
        is_new_segment = False
        
        if '**欢迎进行中医体质检测!**' in result or '下面我们开始进行问诊' in result:
            is_new_segment = True
        
        # 如果当前段不为空，且uu_id发生变化，也认为是新段
        if current_segment and current_uu_id != uu_id:
            is_new_segment = True
        
        if is_new_segment and current_segment:
            # 保存当前段
            segments.append({
                'rows': current_segment,
                'content': '\n'.join([r['result'] for r in current_segment])
            })
            current_segment = []
        
        current_segment.append(row)
        current_uu_id = uu_id
    
    # 添加最后一个段
    if current_segment:
        segments.append({
            'rows': current_segment,
            'content': '\n'.join([r['result'] for r in current_segment])
        })
    
    return segments

def get_content_hash(content):
    """获取内容的哈希值用于去重"""
    # 标准化内容：去除多余空白，统一换行符
    normalized = ' '.join(content.split())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def deduplicate_segments(segments):
    """删除重复的对话段"""
    seen_hashes = {}
    unique_segments = []
    
    for segment in segments:
        content_hash = get_content_hash(segment['content'])
        
        if content_hash not in seen_hashes:
            seen_hashes[content_hash] = True
            unique_segments.append(segment)
    
    return unique_segments

def assign_new_uu_ids(segments):
    """为每段对话分配新的唯一uu_id"""
    result_rows = []
    new_uu_id = 1
    
    for segment in segments:
        for row in segment['rows']:
            row['uu_id'] = str(new_uu_id)
            result_rows.append(row)
        new_uu_id += 1
    
    return result_rows

def write_csv_data(filepath, data, header):
    """写入CSV文件"""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)

def read_remaining_data(filepath, start_row=2078):
    """读取从start_row行开始的所有数据（不包括表头）"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= start_row - 1:  # -1 因为不包括表头
                data.append(row)
    return data

def main():
    input_file = '/home/zkti/fxy/medical-ppo_1/data_5/question.csv'
    output_file = '/home/zkti/fxy/medical-ppo_1/data_5/question.csv'
    backup_file = '/home/zkti/fxy/medical-ppo_1/data_5/question_backup.csv'
    
    # 先备份原文件
    import shutil
    print("正在备份原文件...")
    shutil.copy2(input_file, backup_file)
    print(f"备份已保存到: {backup_file}")
    
    print("正在读取前2077行数据...")
    data = read_csv_data(input_file, max_row=2078)
    print(f"读取了 {len(data)} 行数据")
    
    print("正在识别对话段...")
    segments = identify_conversation_segments(data)
    print(f"识别出 {len(segments)} 个对话段")
    
    print("正在删除重复的对话段...")
    unique_segments = deduplicate_segments(segments)
    print(f"去重后剩余 {len(unique_segments)} 个唯一对话段")
    print(f"删除了 {len(segments) - len(unique_segments)} 个重复对话段")
    
    print("正在分配新的uu_id...")
    processed_data = assign_new_uu_ids(unique_segments)
    print(f"处理后的数据有 {len(processed_data)} 行")
    
    print("正在读取2078行之后的数据...")
    remaining_data = read_remaining_data(input_file, start_row=2078)
    print(f"读取了 {len(remaining_data)} 行后续数据")
    
    print("正在合并数据并写入文件...")
    final_data = processed_data + remaining_data
    header = ['result', 'type', 'uu_id']
    write_csv_data(output_file, final_data, header)
    
    print(f"处理完成！")
    print(f"原始前2077行: {len(data)} 行")
    print(f"处理后前2077行: {len(processed_data)} 行")
    print(f"删除了 {len(data) - len(processed_data)} 行重复数据")
    print(f"后续数据: {len(remaining_data)} 行")
    print(f"最终文件总行数: {len(final_data)} 行")

if __name__ == '__main__':
    main()

