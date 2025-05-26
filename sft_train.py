import torch
import torch.nn as nn
import os
from model.minigpt import MiniLLM
from tokenizer import VOCAB_SIZE, PAD_TOKEN_ID, tokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.sft_dataset import SFTJsonlDataset
from utils.config import Config
from utils.oss_upload import upload_file_to_oss, download_file_from_oss
import platform
import time

cfg = Config()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

sft_dataset = SFTJsonlDataset(cfg.sft_train, tokenizer, seq_len=cfg.max_seq_len)
sft_loader = DataLoader(
    sft_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, drop_last=False
)

model = MiniLLM(
    VOCAB_SIZE, cfg.embed_dim, cfg.max_seq_len,
    cfg.num_heads, num_layers=cfg.num_layers, dropout=cfg.dropout
).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

def load_checkpoint(model, optimizer, filepath):
    if not os.path.exists(filepath):
        download_file_from_oss(filepath, filepath)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'] 

def calc_accuracy(pred_logits, targets):
    preds = pred_logits.argmax(dim=-1)
    mask = targets != PAD_TOKEN_ID
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

def train_sft():
    accum_steps = 16
    last_loss = None
    start_epoch = 0

    # checkpoint逻辑同上
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
        download_file_from_oss(cfg.pre_save_path, cfg.pre_save_path)
        state = torch.load(cfg.pre_save_path, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print("SFT无权重，从头训练")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        count = 0
        optimizer.zero_grad()
        for step, (x, y) in enumerate(sft_loader):
            x = x.to(device)
            y = y.to(device)
            attention_mask = (x != PAD_TOKEN_ID)
            logits = model(x, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss = loss / accum_steps
            loss.backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(sft_loader):
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * accum_steps
            total_acc += calc_accuracy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            count += 1
        total_loss /= count
        total_acc /= count
        scheduler.step(total_loss)
        save_checkpoint(model, optimizer, epoch, cfg.sft_checkpoint_path)
        with open(cfg.sft_log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{total_loss},{total_acc}\n")
        print(f"SFT Epoch {epoch}: Loss {total_loss:.4f} Acc {total_acc:.4f}")
        last_loss = total_loss
    print(f"SFT训练结束，权重已保存到 {cfg.save_path}")
    torch.save(model.state_dict(), cfg.save_path)
    upload_file_to_oss(cfg.save_path, cfg.save_path)

if __name__ == "__main__":
    train_sft()
