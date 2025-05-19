with open("data/common_3000_chars.txt", encoding="utf-8") as f:
    chars = list(f.read().strip())  # 一次性读取所有字，直接转成字符list
VOCAB = chars + ["<unk>", "<pad>"]
word_to_id = {w: i for i, w in enumerate(VOCAB)}
id_to_word = {i: w for w, i in word_to_id.items()}

def tokenize(text):
    # word转token
    return [word_to_id.get(c, word_to_id["<unk>"]) for c in text.lower()]

def detokenize(ids):
    # token转word
    return ''.join([id_to_word.get(i, "?") for i in ids])