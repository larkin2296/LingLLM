import sentencepiece as spm
from transformers import AlbertTokenizerFast

spm.SentencePieceTrainer.train(
    input='corpus.txt',           # 输入文件路径
    model_prefix='spm_bpe',       # 输出文件前缀，生成spm_bpe.model和spm_bpe.vocab
    vocab_size=8000,              # 词汇量（可根据需求调整）
    model_type='bpe',             # 使用BPE算法
    character_coverage=0.9995,    # 覆盖中文常用字符
    user_defined_symbols=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']  # 特殊符号
)

# 加载SentencePiece的model和vocab文件
tokenizer = AlbertTokenizerFast(tokenizer_file="spm_bpe.model")

# 添加特殊Token
tokenizer.add_special_tokens({
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
})
# 保存成HuggingFace标准格式
tokenizer.save_pretrained("hf_tokenizer")

print("训练完成，生成spm_bpe.model和spm_bpe.vocab。")