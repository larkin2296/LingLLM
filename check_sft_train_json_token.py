import json
from collections import Counter
from tokenizer import tokenizer
from utils.config import Config

cfg = Config()

output_token_counter = Counter()

outputs = []

with open(cfg.sft_train, 'r', encoding='utf-8') as fin:
    for line in fin:
        sample = json.loads(line)
        output_text = sample.get("output", "")
        outputs.append(sample.get("output", ""))
        output_token_ids = tokenizer.encode(output_text)
        output_token_counter.update(output_token_ids)

print("output部分最常用的tokens及频次：", output_token_counter.most_common(20))
print("样本总数：", len(outputs))
print("output去重后数量：", len(set(outputs)))
print("平均output长度：", sum(len(x) for x in outputs) / len(outputs))
# print("最常见的20个output：", Counter(outputs).most_common(20))

# 可以反查id->token
for token_id, freq in output_token_counter.most_common(20):
    print(f"{token_id}: {tokenizer.decode([token_id])} -> {freq}")
