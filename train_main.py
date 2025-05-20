import torch
import torch.nn as nn
import os
from model.minigpt import MiniLLM
from tokenizer import tokenize, VOCAB
from dataset import CharDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.read import load_data_from_dir
import csv
import argparse

class Config:
    # 训练集地址
    train_dir = "data/train/"
    # 验证集地址
    val_dir = "data/val/"
    batch_size = 32
    # 最大序列长度
    max_seq_len = 128
    # 它能在训练时随机“丢弃”一部分神经元（设为0），防止模型对训练数据“死记硬背”而失去泛化能力。
    dropout = 0.2
    # 模型维度
    embed_dim = 128
    # 多头注意力机制的头数
    num_heads = 4
    # Transformer Block的层数
    num_layers = 4
    # 训练轮数
    epochs = 70
    # 模型保存路径
    save_path = "weights/minigpt_best.pth"
    checkpoint_path = "weights/checkpoint.pth"

cfg = Config()

train_data = load_data_from_dir(cfg.train_dir)
val_data = load_data_from_dir(cfg.val_dir)
max_seq_len = cfg.max_seq_len
train_dataset = CharDataset(train_data, max_seq_len)
val_dataset = CharDataset(val_data, max_seq_len)
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=2,
    drop_last=False,
    # collate_fn=collate_fn,  # 若做pad再加
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("使用设备:",device)
# print("可用GPU数量：", torch.cuda.device_count())
# print("当前GPU名称：", torch.cuda.get_device_name(0))
vocab_size = len(VOCAB)
embed_dim = cfg.embed_dim
num_heads = cfg.num_heads

model = MiniLLM(
    vocab_size, embed_dim, max_seq_len,
    num_heads, num_layers=cfg.num_layers, dropout=cfg.dropout
).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=VOCAB.index("<pad>"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)

best_val_loss = float('inf')
patience = 3
counter = 0
STOP_THRESHOLD = 0.0005

log_file = "logs/train_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

# 恢复训练
def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']  # 返回恢复的epoch

def calc_accuracy(pred_logits, targets):
    preds = pred_logits.argmax(dim=-1)
    mask = targets != VOCAB.index("<pad>")
    valid_count = mask.sum().item()
    if valid_count == 0:
        return 0.0
    correct = (preds == targets) & mask
    return correct.sum().item() / valid_count

def train(allow_early_stop=False):
    global best_val_loss, counter  # 必须加global，否则会UnboundLocalError
    if os.path.exists(cfg.checkpoint_path):
        print("加载检查点...")
        start_epoch = load_checkpoint(model, optimizer, cfg.checkpoint_path)
        print(f"恢复训练从第 {start_epoch} 轮开始。")
    epochs = cfg.epochs
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_acc += calc_accuracy(logits.view(-1, vocab_size), y.view(-1))
        total_train_loss /= len(train_loader)
        total_train_acc /= len(train_loader)

        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
                total_val_loss += loss.item()
                total_val_acc += calc_accuracy(logits.view(-1, vocab_size), y.view(-1))
        total_val_loss /= len(val_loader)
        total_val_acc /= len(val_loader)

        scheduler.step(total_val_loss)

        print(f"Epoch {epoch}: Train loss: {total_train_loss:.4f}, "
              f"Train acc: {total_train_acc:.4f} | "
              f"Val loss: {total_val_loss:.4f}, "
              f"Val acc: {total_val_acc:.4f}")
        
        # Early stopping
        if total_val_loss < best_val_loss - 1e-5:
            best_val_loss = total_val_loss
            counter = 0
            torch.save(model.state_dict(), cfg.save_path)
        else:
            counter += 1

        if allow_early_stop:
            if total_val_loss < STOP_THRESHOLD or counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if (epoch + 1) % 10 == 0:  # 每10个epoch保存一次
            save_checkpoint(model, optimizer, epoch, cfg.checkpoint_path)

        # 日志
        with open(log_file, "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, total_train_loss, total_train_acc, total_val_loss, total_val_acc])

    print(f"模型权重已保存到 {cfg.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow_early_stop", type=int, default=0)
    args = parser.parse_args()
    allow_early_stop = bool(args.allow_early_stop)
    try:
        train(allow_early_stop=allow_early_stop)
    except KeyboardInterrupt:
        print("训练被手动中断，正在保存模型...")
        save_checkpoint(model, optimizer, epoch, cfg.checkpoint_path)
    print("模型已保存，训练可以在下次继续。")
    print(f"训练数据条数: {len(train_data)}，验证数据条数: {len(val_data)}")
