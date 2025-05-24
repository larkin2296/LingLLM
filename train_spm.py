import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='./data/train/spm_sampled.txt',           # 输入文件路径
    model_prefix='spm_bpe',       # 输出文件前缀，生成spm_bpe.model和spm_bpe.vocab
    vocab_size=32000,              # 词汇量（可根据需求调整）
    model_type='bpe',             # 使用BPE算法
    character_coverage=0.9995,    # 覆盖中文常用字符
    user_defined_symbols=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']  # 特殊符号
)

print("训练完成，生成spm_bpe.model和spm_bpe.vocab。")