import json
import torch
from torch.utils.data import Dataset
from tokenizer import PAD_TOKEN_ID

class SFTJsonlDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, seq_len=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                j = json.loads(line)
                instruction = j.get("instruction", "")
                input_ = j.get("input", "")
                output = j.get("output", "")
                # 这里可根据你的prompt模板格式调整 ↓↓↓
                # 例：Alpaca
                if input_:
                    prompt = f"Instruction: {instruction}\nInput: {input_}\nOutput:"
                else:
                    prompt = f"Instruction: {instruction}\nOutput:"
                prompt_token_ids = tokenizer.encode(prompt)
                output_token_ids = tokenizer.encode(output)
                input_ids = prompt_token_ids + output_token_ids
                labels = [-100] * len(prompt_token_ids) + output_token_ids
                input_ids = input_ids[:seq_len]
                labels = labels[:seq_len]
                if len(input_ids) < seq_len:
                    input_ids += [PAD_TOKEN_ID] * (seq_len - len(input_ids))
                    labels += [-100] * (seq_len - len(labels))
                # target: output部分作为监督目标，可以根据需求mask prompt部分，只监督 output
                # 简单做法：全部训练
                self.samples.append((input_ids, labels))
                
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        # 转tensor
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
