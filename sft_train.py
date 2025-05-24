import torch
import torch.nn as nn
import os
from model.minigpt import MiniLLM
from tokenizer import VOCAB_SIZE, PAD_TOKEN_ID
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import GPT2Tokenizer
from utils.bin_dataset import BinTokenDataset
from utils.config import Config
from utils.oss_upload import upload_file_to_oss
import platform
import time

cfg = Config()

train_dataset = BinTokenDataset(cfg.sft_train_bin, seq_len=cfg.max_seq_len, max_tokens=10_000_000)
max_seq_len = cfg.max_seq_len
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=2,
    drop_last=False,
    # collate_fn=collate_fn,  # 若做pad再加
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available() and platform.system() == "Darwin":
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = MiniLLM(
    VOCAB_SIZE, cfg.embed_dim, cfg.max_seq_len,
    cfg.num_heads, num_layers=cfg.num_layers, dropout=cfg.dropout
).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)

best_loss = float('inf')
patience = 3
counter = 0
STOP_THRESHOLD = 0.0005

if not os.path.exists(cfg.sft_log_file):
    with open(cfg.sft_log_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc\n")

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
    upload_file_to_oss(filepath, filepath)

def train():
    global best_loss, counter
    if os.path.exists(cfg.checkpoint_path):
        print("加载SFT检查点...")
        checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"恢复训练从第 {start_epoch} 轮开始。")
    # 2. 没有SFT checkpoint就加载预训练权重
    elif os.path.exists(cfg.pre_save_path):
        print("加载预训练权重 weights/minigpt_best.pth ...")
        state = torch.load("weights/minigpt_best.pth", map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print("加载完毕！")
        start_epoch = 0
    else:
        print("未找到预训练权重，将随机初始化参数！")
        start_epoch = 0
    if os.path.exists(cfg.checkpoint_path):
        print("加载检查点...")
        checkpoint = torch.load(cfg.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"恢复训练从第 {start_epoch} 轮开始。")
    epochs = cfg.epochs
    for epoch in range(epochs):
        epoch_start_time = time.time()  # 记录epoch开始时间

        model.train()
        total_loss = 0
        total_acc = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += calc_accuracy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        total_loss /= len(train_loader)
        total_acc /= len(train_loader)

        scheduler.step(total_loss)

        epoch_seconds = time.time() - epoch_start_time

        print(f"Pre-training Epoch {epoch}: Loss: {total_loss:.4f}, Acc: {total_acc:.4f}| Time: {epoch_seconds:.2f}S")

        if total_loss < best_loss - 1e-5:
            best_loss = total_loss
            counter = 0
            torch.save(model.state_dict(), cfg.save_path)
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, cfg.checkpoint_path)

        with open(cfg.sft_log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{total_loss},{total_acc}\n")

    print(f"模型权重已保存到 {cfg.save_path}")
    torch.save(model.state_dict(), cfg.save_path)
    upload_file_to_oss(cfg.save_path, cfg.save_path)

if __name__ == "__main__":
    train()
