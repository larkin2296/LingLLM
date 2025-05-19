import torch
from torch.utils.data import Dataset
from tokenizer import VOCAB

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_seq_len, vocab=VOCAB, pad_token="<pad>", unk_token="<unk>"):
        self.data = data
        self.max_seq_len = max_seq_len
        self.vocab = vocab
        self.pad_idx = vocab.index(pad_token)
        self.unk_idx = vocab.index(unk_token)

    def text2idx(self, text):
        # 字符串 -> id list
        return [self.vocab.index(c) if c in self.vocab else self.unk_idx for c in text]

    def pad(self, ids):
        # 截断+补pad
        ids = ids[:self.max_seq_len]
        ids += [self.pad_idx] * (self.max_seq_len - len(ids))
        return ids

    def __getitem__(self, idx):
        line = self.data[idx].strip()
        # 1. 问答对（带tab）
        if '\t' in line:
            src, tgt = line.split('\t', 1)
            x = self.text2idx(src)
            y = self.text2idx(tgt)
        # 2. 单句或一段，做自回归（常用于文章、无标签语料）
        else:
            x = self.text2idx(line)
            # 自回归：用输入预测下一个token
            # y为x的“右移一位”，最后一位为pad
            y = x[1:] + [self.pad_idx]
            # x本身右侧补pad
        x = self.pad(x)
        y = self.pad(y)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.data)
