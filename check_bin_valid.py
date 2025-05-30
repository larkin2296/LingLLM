import numpy as np
arr = np.memmap('./data/train/pre_train_token_ids.bin', dtype=np.int32, mode='r')
print(arr[:100])