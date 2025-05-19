import torch
from model.minigpt import MiniLLM
from tokenizer import tokenize, detokenize, VOCAB

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MiniLLM(len(VOCAB), embed_dim=64, max_seq_len=64, num_heads=4, num_layers=2, dropout=0.1).to(device)
model.load_state_dict(torch.load("weights/minigpt_best.pth", map_location=device))
model.eval()

def infer_one_sample(text, max_gen=64):
    """
    给定模型和输入文本，生成模型的回复。
    model: 已加载权重的MiniLLM
    text: str，输入文本
    max_gen: 最多生成多少个字符
    device: 推理设备
    """
    model.eval()
    # tokenize 输入
    input_ids = tokenize(text)
    input_ids = input_ids[:model.max_seq_len-1]  # 截断，保证不超过模型长度
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 生成用的列表，初始是输入
    generated = input_ids.copy()
    
    for _ in range(max_gen):
        # 只保留max_seq_len长度
        curr_input = torch.tensor([generated[-model.max_seq_len:]], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(curr_input)  # [1, seq, vocab_size]
        # 取最后一个位置的输出
        last_logits = logits[0, -1]  # [vocab_size]
        pred_id = last_logits.argmax().item()
        generated.append(pred_id)
        # 假设 <pad> 或 <eos> 是回复终止标记
        if VOCAB[pred_id] in ["<pad>", "<eos>"]:
            break

    # 返回生成的内容（去掉输入部分，只保留回复部分）
    reply_ids = generated[len(input_ids):]
    reply = detokenize(reply_ids)
    return reply
