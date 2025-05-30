import sentencepiece as spm
from tokenizer import PAD_TOKEN_ID
sp = spm.SentencePieceProcessor()
sp.load('spm_bpe.model')

test = "测试一下中文分词"
ids = sp.encode(test, out_type=int)
print("编码:", ids)
print("解码:", sp.decode(ids))

print("pad_id:", sp.pad_id())       # -1 说明没pad
print("unk_id:", sp.unk_id())       # 0
print("bos_id:", sp.bos_id())       # -1 (默认没加)
print("eos_id:", sp.eos_id())       # -1
print("id_to_piece(0):", sp.id_to_piece(0))   # <unk>
print("你的pad_id:", sp.pad_id())
print("你训练时用的PAD_TOKEN_ID:", PAD_TOKEN_ID)
for sym in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
    print(f"{sym}: id={sp.piece_to_id(sym)}")