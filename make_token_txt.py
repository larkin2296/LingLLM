from datasets import load_dataset
import sentencepiece as spm
import numpy as np
from tqdm import tqdm

# 配置
DATA_FILE = "train.txt"  # 你的文本文件名
OUTPUT_BIN = "train_token_ids.bin"
SENTENCEPIECE_MODEL = "spm_bpe.model"
ADD_EOS = True

# 加载分词器
sp = spm.SentencePieceProcessor()
sp.load(SENTENCEPIECE_MODEL)
print("分词器 vocab_size =", sp.get_piece_size())

# 加载数据集
dataset = load_dataset("text", data_files={"train": DATA_FILE})["train"]
print("总样本数:", len(dataset))

with open(OUTPUT_BIN, "wb") as fout:
    max_id, min_id, total = 0, 1 << 30, 0
    for example in tqdm(dataset, desc="分词写bin"):
        text = example["text"].strip()
        if not text: continue
        ids = sp.encode(text, out_type=int)
        if ADD_EOS:
            ids.append(sp.eos_id())
        arr = np.array(ids, dtype=np.int32)
        arr.tofile(fout)
        if arr.size > 0:
            max_id = max(max_id, arr.max())
            min_id = min(min_id, arr.min())
            total += arr.size
    print(f"\n已保存: {OUTPUT_BIN}")
    print(f"Token总数: {total}")
    print(f"Token id范围: {min_id} ~ {max_id}")
    print(f"建议vocab_size >= {max_id + 1}")
