# 假设两个文件都在当前目录下
dialogs_file = './data/dialogs.txt'
chars_file = './data/common_3000_chars.txt'

# 读取常用字表，存入集合（去重，便于查找）
with open(chars_file, 'r', encoding='utf-8') as f:
    common_chars = set(f.read().strip())

# 读取 dialogs.txt 里的所有文本
with open(dialogs_file, 'r', encoding='utf-8') as f:
    dialogs_content = f.read()

# 检查每个字是否在常用字集合里
uncommon_chars = set()
for ch in dialogs_content:
    if ch not in common_chars and ch.strip():  # 排除空白和标点
        uncommon_chars.add(ch)

if uncommon_chars:
    print("dialogs.txt 中有以下字不在 common_3000_chars.txt 里：")
    print(''.join(sorted(uncommon_chars)))
else:
    print("dialogs.txt 中所有汉字都在 common_3000_chars.txt 里。")
