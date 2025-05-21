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
        # 兼容两种格式
        # 格式1: {"instruction", "input", "output"}
        if "instruction" in item and "output" in item:
            # 拼接input，如果有input则instruction+input，否则只有instruction
            if item.get("input", ""):
                src = item["instruction"].strip() + "\n" + item["input"].strip()
            else:
                src = item["instruction"].strip()
            tgt = item["output"].strip()
        # 格式2: {"question", "answer"}
        elif "question" in item and "answer" in item:
            src = item["question"].strip()
            tgt = item["answer"].strip()
        else:
            raise ValueError(f"Unknown data format: {item}")

        x = self.tokenizer.encode(src, add_special_tokens=False, max_length=self.max_seq_len, truncation=True)
        y = self.tokenizer.encode(tgt, add_special_tokens=False, max_length=self.max_seq_len, truncation=True)
        x = self.pad(x)
        y = self.pad(y)

        assert len(x) == self.max_seq_len, f"input长度不对: {len(x)}"
        assert len(y) == self.max_seq_len, f"label长度不对: {len(y)}"

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.data)
