import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='./data/train/spm_sampled.txt',
    model_prefix='spm_bpe',
    vocab_size=32000,
    model_type='bpe',
    character_coverage=0.9995,
    pad_id=0,        # 让0号token是真正pad
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=['[CLS]', '[SEP]', '[MASK]']
)

print("训练完成，生成spm_bpe.model和spm_bpe.vocab。")