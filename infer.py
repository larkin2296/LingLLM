import torch
from model.minigpt import MiniLLM
from tokenizer import tokenize, detokenize, VOCAB

# 模型参数与训练时保持一致
vocab_size = len(VOCAB)
embed_dim = 16
max_seq_len = 32
num_heads = 2

# 新建模型，加载已训练参数（如果有保存的话）
model = MiniLLM(vocab_size, embed_dim, max_seq_len, num_heads)
model.load_state_dict(torch.load("minigpt.pth"))  # 如果你训练后保存了模型权重

model.eval()

sentence = "hello "
token_ids = tokenize(sentence)

input_ids = torch.tensor([token_ids], dtype=torch.long)
with torch.no_grad():
    logits = model(input_ids)
    next_token_logits = logits[0, -1, :]
    next_token_id = next_token_logits.argmax().item()
    print(f"输入: {sentence}，模型预测下一个字符是: {VOCAB[next_token_id]}")
