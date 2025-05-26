import json
import sentencepiece as spm
from tqdm import tqdm

# 配置
IN_FILE = "data/val/distill_r1_110k_sft.jsonl"           # 输入SFT jsonl
OUT_FILE = "data/val/sft_data.filtered.jsonl" # 输出clean版
SP_MODEL = "spm_bpe.model"                      # 你的分词器
MAX_SEQ_LEN = 512

# 加载分词器
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL)

total, keep, drop, truncated = 0, 0, 0, 0

with open(IN_FILE, "r", encoding="utf-8") as fin, \
     open(OUT_FILE, "w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="过滤SFT样本"):
        try:
            j = json.loads(line)
            instruction = j.get("instruction", "")
            input_ = j.get("input", "")
            output = j.get("output", "")
            prefix = f"Instruction: {instruction}\nInput: {input_}\nOutput:"
            prefix_ids = sp.encode(prefix)
            max_output_len = MAX_SEQ_LEN - len(prefix_ids)
            if max_output_len <= 0:
                drop += 1
                continue
            output_ids = sp.encode(output)[:max_output_len]
            total_ids = prefix_ids + output_ids
            if len(total_ids) <= MAX_SEQ_LEN:
                if len(sp.encode(output)) > max_output_len:
                    truncated += 1
                j["output"] = sp.decode(output_ids)
                fout.write(json.dumps(j, ensure_ascii=False) + "\n")
                keep += 1
            else:
                drop += 1
            total += 1
        except Exception as e:
            drop += 1
            continue

print(f"总数: {total}，保留: {keep}（其中被截断output的有: {truncated})，剔除: {drop}，保留率: {keep/total:.2%}, ，其中{truncated}条output发生了截断。")
