# import numpy as np
# arr = np.memmap('./data/train/pre_train_token_ids.bin', dtype=np.int32, mode='r')
# print(arr[:100])
# print('0的数量:', (arr == 0).sum())
# print('非0的数量:', (arr != 0).sum())
# print('最小值:', arr.min())
# print('最大值:', arr.max())
import json
import random

path = './data/train/mobvoi_seq_monkey_general_open_corpus.jsonl'
total = sum(1 for _ in open(path, encoding='utf-8'))
samples = random.sample(range(total), 5)

with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i in samples:
            obj = json.loads(line)
            print(obj)