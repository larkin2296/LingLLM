import numpy as np

# 假设每个token是int32
arr = np.fromfile('data/train/train.bin', dtype=np.int64)
print('max token id:', arr.max())
print('min token id:', arr.min())
print('推荐vocab_size:', arr.max() + 1)