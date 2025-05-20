import torch
from torch.utils.data import Dataset
import json

class QAJsonlDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []

        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    self.data.append(obj)
        self.pad_idx = getattr(tokenizer, "pad_token_id", 0)

    def pad(self, ids):
        ids = ids[:self.max_seq_len]
        ids += [self.pad_idx] * (self.max_seq_len - len(ids))
        return ids

    def __getitem__(self, idx):
        item = self.data[idx]
        x = self.tokenizer.encode(item["question"], add_special_tokens=False)
        y = self.tokenizer.encode(item["answer"], add_special_tokens=False)
        x = self.pad(x)
        y = self.pad(y)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.data)
