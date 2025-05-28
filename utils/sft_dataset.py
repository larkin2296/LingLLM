import json
import torch
from torch.utils.data import Dataset
from tokenizer import PAD_TOKEN_ID

class SFTJsonlDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, seq_len=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        half_seq_len = seq_len // 2
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                j = json.loads(line)
                prompt = j.get("question", "")
                output = j.get("answer", "")
                # 这里可根据你的prompt模板格式调整 ↓↓↓
                # 例：Alpaca
                prompt_token_ids = tokenizer.encode(prompt)
                output_token_ids = tokenizer.encode(output)
                if len(prompt_token_ids) < half_seq_len:
                    input_ids = (prompt_token_ids + output_token_ids)[:seq_len]
                    labels = [-100] * len(prompt_token_ids) + output_token_ids
                    labels = labels[:seq_len]
                    if len(input_ids) < seq_len:
                        input_ids += [PAD_TOKEN_ID] * (seq_len - len(input_ids))
                        labels += [-100] * (seq_len - len(labels))
                else:
                    new_prompt_token_ids = prompt_token_ids[:half_seq_len]
                    input_ids = new_prompt_token_ids + output_token_ids
                    labels = [-100] * len(new_prompt_token_ids) + output_token_ids
                    input_ids = input_ids[:seq_len]
                    labels = labels[:seq_len]
                    if len(input_ids) < seq_len:
                        input_ids += [PAD_TOKEN_ID] * (seq_len - len(input_ids))
                        labels += [-100] * (seq_len - len(labels))
                self.samples.append((input_ids, labels))
                print(f"len(prompt_token_ids): {len(prompt_token_ids)}, len(output_token_ids): {len(output_token_ids)}, output tokens (不为-100) count: {sum(x != -100 for x in labels)}")
                
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        # 转tensor
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
