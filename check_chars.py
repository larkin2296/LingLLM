import os

# 假设常用字文件路径
chars_file = './data/common_3000_chars.txt'
train_dir = "data/train/"

# 读取常用字表，存入集合（去重，便于查找）
with open(chars_file, 'r', encoding='utf-8') as f:
    common_chars = set(f.read().strip())

# 获取data/train/文件夹下的所有文件
train_files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]

# 用于保存不在常用字集合中的字符
uncommon_chars = set()

# 用于记录每一行的长度
line_lengths = []

# 逐个文件读取并检查
for file_name in train_files:
    file_path = os.path.join(train_dir, file_name)
    
    # 打开每个文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        # 遍历文件中的每一行
        for line in lines:
            # 将每一行按 tab 分割为问题和答案
            question, answer = line.strip().split("\t")

            # 计算问题和答案的总长度
            line_length = len(question + answer)
            line_lengths.append(line_length)

            # 检查问题和答案中的每个字符是否在常用字集合中
            for ch in question + answer:  # 合并问题和答案进行检查
                if ch not in common_chars and ch.strip():  # 排除空白字符
                    uncommon_chars.add(ch)

# 输出不在常用字集合中的字符
if uncommon_chars:
    print("以下字不在 common_3000_chars.txt 里：")
    print(''.join(sorted(uncommon_chars)))
else:
    print("所有汉字都在 common_3000_chars.txt 里。")

# 计算文本的最大长度（或选择一个百分位数）
max_line_length = max(line_lengths)
print(f"数据集中的最大文本长度为：{max_line_length} 个字符")

# 你可以选择百分位数（例如 90% 的文本长度以内）
import numpy as np
percentile_90 = np.percentile(line_lengths, 90)
print(f"90%的文本长度以内的最大长度为：{percentile_90} 个字符")

# 设定合适的 max_seq_len（你可以根据数据情况调整）
max_seq_len = int(percentile_90)  # 选择 90% 文本长度的最大值
print(f"建议的最大序列长度为：{max_seq_len}")
