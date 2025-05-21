import torch
from model.minigpt import MiniLLM
from transformers import GPT2Tokenizer
from tokenizer import encode, decode, PAD_TOKEN_ID
from utils.config import Config

cfg = Config()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
VOCAB_SIZE = len(tokenizer)
PAD_TOKEN_ID = tokenizer.pad_token_id

# === 2. 新建模型并加载训练权重 ===
model = MiniLLM(VOCAB_SIZE, cfg.embed_dim,cfg.max_seq_len, cfg.num_heads, cfg.num_layers)
state = torch.load("weights/sft_minigpt_best.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

# === 3. 对话循环 ===
while True:
    prompt = input("\n你说: ").strip()
    if prompt.lower() in ['exit', 'quit', 'q']:
        print("再见！")
        break
    # 4. 用分词器转成token id
    input_ids = encode(prompt)
    if len(input_ids) > cfg.max_seq_len - 1:
        input_ids = input_ids[-(cfg.max_seq_len-1):]  # 保证长度不超限制

    # 5. 循环生成回复，直到遇到<pad>或到最大长度
    generated = input_ids.copy()
    for _ in range(50):  # 最多生成50个字，可调
        input_tensor = torch.tensor([generated[-(cfg.max_seq_len-1):]], dtype=torch.long)
        logits = model(input_tensor)
        next_token_logits = logits[0, -1, :]
        next_token_id = next_token_logits.argmax().item()  # 选概率最大token

        if next_token_id == PAD_TOKEN_ID:
            break

        generated.append(next_token_id)

        next_token_str = decode([next_token_id])
        if next_token_str in ['。', '?', '？', '！', '!']:
            break

    # 6. 去掉输入内容，只输出生成的回复
    reply_token_ids = generated[len(input_ids):]
    reply = decode(reply_token_ids)
    print(f"MiniLLM: {reply}")
