import torch
from torch.utils.data import Dataset
import json
from tokenizer import PAD_TOKEN_ID, encode, piece

class SFTDataset(Dataset):
    def __init__(self, jsonl_file, max_seq_len=512):
        self.samples = []
        self.max_seq_len = max_seq_len
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                instruction = j.get("instruction", "")
                input_ = j.get("input", "")
                output = j.get("output", "")
                # 构造 SFT prompt
                prompt = f"Instruction: {instruction}\nInput: {input_}\nOutput: "
                prompt_ids = encode(prompt)
                output_ids = encode(output)
                input_ids = prompt_ids + output_ids
                # 截断
                input_ids = input_ids[:max_seq_len]
                # label: input/prompt部分为-100，output部分真实target
                labels = [-100] * len(prompt_ids) + output_ids
                labels = labels[:max_seq_len]
                # 对齐
                if len(input_ids) < max_seq_len:
                    pad_len = max_seq_len - len(input_ids)
                    input_ids += [PAD_TOKEN_ID] * pad_len
                    labels += [-100] * pad_len
                assert all(0 <= i < piece() for i in input_ids), f"非法id in {input_ids}, max={max(input_ids)}, vocab={piece()}"
                self.samples.append((input_ids, labels))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)
    # def __getitem__(self, idx):
    #     x, y = self.samples[idx]
    #     attention_mask = [int(i != PAD_TOKEN_ID) for i in x]
    #     return torch.tensor(x), torch.tensor(y), torch.tensor(attention_mask)
