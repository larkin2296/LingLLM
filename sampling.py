import random, json

src = './data/train/mobvoi_seq_monkey_general_open_corpus.jsonl'
dst = './data/train/spm_sampled.txt'
sample_size = 1000000  # 你想采多少条

result = []
total = 0

with open(src, 'r', encoding='utf-8') as fin:
    for line in fin:
        try:
            text = json.loads(line)["text"]  # 只采 text 字段
            total += 1
            if len(result) < sample_size:
                result.append(text)
            else:
                j = random.randint(0, total - 1)
                if j < sample_size:
                    result[j] = text
        except Exception:
            continue

print(f"实际采样行数: {len(result)}，原始总行数: {total}")
with open(dst, 'w', encoding='utf-8') as fout:
    for l in result:
        one_line = l.replace('\n', ' ').replace('\r', ' ')
        fout.write(one_line.strip() + '\n')
print(f"采样完成，生成{dst}")
