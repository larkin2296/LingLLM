import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, data, max_seq_len, tokenizer, pad_token="<pad>", unk_token="<unk>"):
        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        # 取 pad_token_id，如果没有则补 0
        self.pad_idx = getattr(tokenizer, "pad_token_id", 0)
        self.unk_idx = getattr(tokenizer, "unk_token_id", 0)

    def text2idx(self, text):
        # 用外部 tokenizer 编码
        return self.tokenizer.encode(text, add_special_tokens=False)

    def pad(self, ids):
        # 截断+补pad
        ids = ids[:self.max_seq_len]
        ids += [self.pad_idx] * (self.max_seq_len - len(ids))
        return ids

    def __getitem__(self, idx):
        line = self.data[idx].strip()
        if '\t' in line:
            src, tgt = line.split('\t', 1)
            x = self.text2idx(src)
            y = self.text2idx(tgt)
        else:
            x = self.text2idx(line)
            y = x[1:] + [self.pad_idx]
        x = self.pad(x)
        y = self.pad(y)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.data)
