import torch
from torch.utils.data import Dataset

class BinTokenDataset(Dataset):
    def __init__(self, path, seq_len, max_tokens=None):
        import numpy as np
        raw_data = np.memmap(path, dtype=np.uint32, mode='r')
        if max_tokens:
            raw_data = raw_data[:max_tokens]
        self.data = raw_data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len - 1

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        x = torch.from_numpy(self.data[start:end-1].copy()).long()
        y = torch.from_numpy(self.data[start+1:end].copy()).long()
        return x, y

