from datasets import load_dataset
import sentencepiece as spm
import numpy as np
from tqdm import tqdm
import os

# ==== 配置 ====
DATASET_NAME = "C:/Users/54646/Desktop/openwebtextSample/train"
SPLIT = "train"  # 或 "validation"
SENTENCEPIECE_MODEL = "spm_bpe.model"  # 你的分词模型
OUTPUT_BIN = "train_token_ids.bin"
BATCH_SIZE = 4096  # 每次批处理多少条文本，加快encode速度
ADD_EOS = True     # 是否每句加结尾符
# ===============

# 加载 SentencePiece 分词器
sp = spm.SentencePieceProcessor()
sp.load(SENTENCEPIECE_MODEL)
vocab_size = sp.get_piece_size()
print(f"Loaded spm_bpe.model, vocab_size={vocab_size}")

# 加载数据集（只读文本字段）
print(f"Loading dataset {DATASET_NAME} split={SPLIT} ...")
arrow_files = [os.path.join(DATASET_NAME, f) for f in os.listdir(DATASET_NAME) if f.endswith(".arrow")]
arrow_files.sort()
print("找到train下的arrow文件：", arrow_files[:3], "...")  # 只展示前三个

dataset = load_dataset("arrow", data_files={"train": arrow_files})["train"]
print(dataset)
print(dataset[0])
print(f"Total samples: {len(dataset)}")

# 写入模式二进制文件
with open(OUTPUT_BIN, "wb") as fout:
    max_id = 0
    min_id = 1 << 30
    total = 0

    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Encoding & Writing"):
        batch = dataset.select(range(i, min(i + BATCH_SIZE, len(dataset))))
        lines = [item["text"].strip() for item in batch]
        # 批量分词
        ids_batch = [sp.encode(line.strip(), out_type=int) for line in lines]
        # 是否加 EOS
        if ADD_EOS:
            ids_batch = [ids + [sp.eos_id()] for ids in ids_batch]

        # 展平成一维 token id
        flat_ids = [id for ids in ids_batch for id in ids]
        if not flat_ids:
            continue

        arr = np.array(flat_ids, dtype=np.int32)
        arr.tofile(fout)
        max_id = max(max_id, arr.max())
        min_id = min(min_id, arr.min())
        total += arr.size

    print(f"\n已保存为: {OUTPUT_BIN}")
    print(f"Token总数: {total}")
    print(f"Token id最小值: {min_id}  最大值: {max_id}")
    print(f"建议vocab_size >= {max_id + 1}")

