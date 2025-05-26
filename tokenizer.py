import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm_bpe.model")

# tokens = sp.encode(text, out_type=str)

VOCAB_SIZE = sp.get_piece_size()
PAD_TOKEN_ID = 0
tokenizer = sp
def encode(text):
    return sp.encode(text, out_type=int)

def decode(ids):
    return sp.decode(ids)