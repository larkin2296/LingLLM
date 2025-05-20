import torch
from model.minigpt import MiniLLM
from tokenizer import tokenize, detokenize, VOCAB

# === 1. 模型参数保持和训练时一致 ===
vocab_size = len(VOCAB)
embed_dim = 64
max_seq_len = 64
num_heads = 4

# === 2. 新建模型并加载训练权重 ===
model = MiniLLM(vocab_size, embed_dim, max_seq_len, num_heads, num_layers=2)
model.load_state_dict(torch.load("weights/minigpt_best.pth"))
model.eval()

# === 3. 对话循环 ===
while True:
    prompt = input("\n你说: ").strip()
    if prompt.lower() in ['exit', 'quit', 'q']:
        print("再见！")
        break
    # 4. 用分词器转成token id
    input_ids = tokenize(prompt)
    if len(input_ids) > max_seq_len - 1:
        input_ids = input_ids[-(max_seq_len-1):]  # 保证长度不超限制

    # 5. 循环生成回复，直到遇到<pad>或到最大长度
    generated = input_ids.copy()
    for _ in range(50):  # 最多生成50个字，可调
        input_tensor = torch.tensor([generated[-(max_seq_len-1):]], dtype=torch.long)
        logits = model(input_tensor)
        next_token_logits = logits[0, -1, :]
        next_token_id = next_token_logits.argmax().item()  # 选概率最大token

        # 如果生成了<pad>，就停
        if next_token_id == VOCAB.index("<pad>"):
            break
        generated.append(next_token_id)
        # 如果碰到句号或问号，可以提前结束
        if VOCAB[next_token_id] in ['。', '?', '？', '！', '!']:
            break

    # 6. 去掉输入内容，只输出生成的回复
    reply_token_ids = generated[len(input_ids):]
    reply = detokenize(reply_token_ids)
    print(f"MiniLLM: {reply}")
