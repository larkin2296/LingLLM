import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.minigpt2 import TransformerLM
from tokenizer import VOCAB_SIZE, PAD_TOKEN_ID, encode, tokenizer
from utils.sft_dataset import SFTJsonlDataset
from utils.config import Config
from utils.oss_upload import upload_file_to_oss, download_file_from_oss

cfg = Config()

def make_prompt(user_input):
    return user_input
    # return f"Instruction: {user_input}\nOutput:"

def calc_accuracy(pred_logits, targets):
    preds = pred_logits.argmax(dim=-1)
    mask = targets != -100
    valid_count = mask.sum().item()
    if valid_count == 0:
        return 0.0
    correct = (preds == targets) & mask
    return correct.sum().item() / valid_count

def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({ 
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

def load_checkpoint(model, optimizer, filepath):
    if not os.path.exists(filepath):
        download_file_from_oss(filepath, filepath)
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'] 

def topk_generate(model, tokenizer, prompt, k=5, max_new_tokens=64):
    model.eval()
    input_ids = encode(make_prompt(prompt), max_length=cfg.max_seq_len)
    generated = input_ids.copy()
    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([generated[-cfg.max_seq_len:]], dtype=torch.long).to(cfg.device)
        attention_mask = (input_tensor != PAD_TOKEN_ID).long()
        logits = model(input_tensor, attention_mask=attention_mask)
        next_token_logits = logits[0, -1, :]
        topk_values, topk_ids = torch.topk(next_token_logits, k)
        probs = torch.softmax(topk_values, dim=0)
        sampled = torch.multinomial(probs, 1).item()
        next_token_id = topk_ids[sampled].item()
        if next_token_id == PAD_TOKEN_ID:
            break
        generated.append(next_token_id)
    return tokenizer.decode(generated[len(input_ids):])

def evaluate(model, eval_loader, loss_fn):
    model.eval()
    eval_loss, eval_acc, eval_count = 0, 0, 0
    with torch.no_grad():
        for x, y in eval_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            attention_mask = (x != PAD_TOKEN_ID)
            logits = model(x, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
            eval_loss += loss.item()
            eval_acc += calc_accuracy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            eval_count += 1
    eval_loss /= eval_count
    eval_acc /= eval_count
    return eval_loss, eval_acc

def periodic_sample(model, tokenizer, prompts, k=5, max_new_tokens=64):
    print("=" * 50)
    print("SFT定期采样对话：")
    for prompt in prompts:
        output = topk_generate(model, tokenizer, prompt, k=k, max_new_tokens=max_new_tokens)
        print(f"Prompt: {prompt}\nOutput: {output}\n")
    print("=" * 50)

def get_data_chunks(total_len, chunk_size):
    num_chunks = (total_len + chunk_size - 1) // chunk_size
    return [range(i * chunk_size, min((i + 1) * chunk_size, total_len)) for i in range(num_chunks)]

def train_sft():
    # 设备选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.device = device  # 配置可被采样等复用

    sft_dataset = SFTJsonlDataset(cfg.sft_train, tokenizer, seq_len=cfg.max_seq_len)
    eval_dataset = SFTJsonlDataset(cfg.sft_eval, tokenizer, seq_len=cfg.max_seq_len)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, drop_last=False)

    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=cfg.embed_dim,
        n_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.embed_dim*4,
        max_len=cfg.max_seq_len,
        dropout=cfg.dropout
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # 分块采样准备
    total_len = len(sft_dataset)
    chunk_size = 10000
    chunks = get_data_chunks(total_len, chunk_size)
    num_chunks = len(chunks)

    eval_interval = cfg.eval_interval if hasattr(cfg, "eval_interval") else 2
    sample_interval = cfg.sample_interval if hasattr(cfg, "sample_interval") else 2
    prompts = getattr(cfg, "sample_prompts", [
        "你爱阅读吗？",
        "你有什么兴趣爱好？", "上海的别称是什么？"
    ])

    last_loss = None
    start_epoch = 0

    # Checkpoint/权重加载
    if os.path.exists(cfg.sft_checkpoint_path):
        print("加载SFT检查点...")
        start_epoch = load_checkpoint(model, optimizer, cfg.sft_checkpoint_path)
        print(f"恢复训练从第 {start_epoch} 轮开始。")
    elif os.path.exists(cfg.save_path):
        print("加载SFT模型参数（无optimizer）...")
        state = torch.load(cfg.save_path, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print("SFT参数已加载，optimizer和epoch将从头初始化")
    else:
        if not os.path.exists(cfg.pre_save_path):
            download_file_from_oss(cfg.pre_save_path, cfg.pre_save_path)
        state = torch.load(cfg.pre_save_path, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print("SFT无权重，从头训练")

    try:
        for epoch in range(start_epoch, cfg.epochs):
            chunk_idx = epoch % num_chunks
            train_sampler = SubsetRandomSampler(chunks[chunk_idx])
            sft_loader = DataLoader(
                sft_dataset, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=2, drop_last=False
            )

            model.train()
            total_loss, total_acc, count = 0, 0, 0
            epoch_start_time = time.time()
            for step, (x, y) in enumerate(sft_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    attention_mask = (x != PAD_TOKEN_ID)
                    logits = model(x, attention_mask=attention_mask)
                    loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_acc += calc_accuracy(logits.view(-1, VOCAB_SIZE), y.view(-1))
                count += 1
            total_loss /= count
            total_acc /= count
            scheduler.step(total_loss)
            epoch_seconds = time.time() - epoch_start_time
            delta_loss = abs(last_loss - total_loss) if last_loss is not None else 0

            print(f"[TRAIN] Epoch {epoch}: Loss {total_loss:.4f} Acc {total_acc:.4f}| Time: {epoch_seconds:.2f}s| Loss下降: {delta_loss:.6f}")

            save_checkpoint(model, optimizer, epoch, cfg.sft_checkpoint_path)

            # 定期Eval
            if (epoch + 1) % eval_interval == 0:
                eval_loss, eval_acc = evaluate(model, eval_loader, loss_fn)
                print(f"[EVAL]  Epoch {epoch}: Eval Loss {eval_loss:.4f} Eval Acc {eval_acc:.4f}")

            # 定期采样
            if (epoch + 1) % sample_interval == 0:
                periodic_sample(model, tokenizer, prompts, k=5, max_new_tokens=64)

            # 写日志
            with open(cfg.sft_log_file, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{total_loss},{total_acc}\n")
            last_loss = total_loss

            # 手工终止
            if os.path.exists("stop.txt"):
                print("检测到 stop.txt，终止训练。")
                break

    except KeyboardInterrupt:
        print("收到中断信号，优雅退出...")

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.save(model.state_dict(), cfg.save_path)
        print(f"SFT训练结束，权重已保存到 {cfg.save_path}")
        upload_file_to_oss(cfg.save_path, cfg.save_path)

if __name__ == "__main__":
    train_sft()