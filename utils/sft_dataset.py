import torch
from torch.utils.data import Dataset
import json

class SFTDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_seq_len=512):
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
                prompt_ids = tokenizer.encode(prompt, out_type=int)
                output_ids = tokenizer.encode(output, out_type=int)
                input_ids = prompt_ids + output_ids
                # 截断
                input_ids = input_ids[:max_seq_len]
                # label: input/prompt部分为-100，output部分真实target
                labels = [-100] * len(prompt_ids) + output_ids
                labels = labels[:max_seq_len]
                # 对齐
                if len(input_ids) < max_seq_len:
                    pad_len = max_seq_len - len(input_ids)
                    input_ids += [tokenizer.pad_id()] * pad_len
                    labels += [-100] * pad_len
                self.samples.append((input_ids, labels))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)
