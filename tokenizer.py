from transformers import GPT2Tokenizer
from utils.config import Config
cfg = Config()
# 只下载一次即可，之后可指定 cache_dir/local_files_only
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

VOCAB_SIZE = tokenizer.vocab_size
PAD_TOKEN_ID = tokenizer.pad_token_id if tokenizer.pad_token is not None else 0  # 如果没有pad，可以手动指定

def encode(text):
    return tokenizer.encode(text, max_length=cfg.max_seq_len, truncation=True, padding='max_length')

def decode(token_ids):
    return tokenizer.decode(token_ids)