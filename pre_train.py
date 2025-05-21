import torch
import torch.nn as nn
import os
from model.minigpt import MiniLLM
from tokenizer import VOCAB_SIZE, PAD_TOKEN_ID
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.bin_dataset import BinTokenDataset
from utils.config import Config
from utils.oss_upload import upload_file_to_oss, download_file_from_oss

cfg = Config()

train_dataset = BinTokenDataset(cfg.train_bin, seq_len=cfg.max_seq_len, max_tokens=10_000_000)
max_seq_len = cfg.max_seq_len
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=2,
    drop_last=False,
    # collate_fn=collate_fn,  # 若做pad再加
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("使用设备:",device)
# print("可用GPU数量：", torch.cuda.device_count())
# print("当前GPU名称：", torch.cuda.get_device_name(0))
vocab_size = VOCAB_SIZE
embed_dim = cfg.embed_dim
num_heads = cfg.num_heads

model = MiniLLM(
    vocab_size, embed_dim, max_seq_len,
    num_heads, num_layers=cfg.num_layers, dropout=cfg.dropout
).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
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
    upload_file_to_oss(f"oss/{filepath}", filepath)

# 恢复训练
def load_checkpoint(model, optimizer, filepath):
    if not os.path.exists(filepath):
        download_file_from_oss(f"oss/{filepath}", filepath)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']  # 返回恢复的epoch

def calc_accuracy(pred_logits, targets):
    preds = pred_logits.argmax(dim=-1)
    mask = targets != PAD_TOKEN_ID
    valid_count = mask.sum().item()
    if valid_count == 0:
        return 0.0
    correct = (preds == targets) & mask
    return correct.sum().item() / valid_count

def train():
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

        print(f"Pre-training Epoch {epoch}: Loss: {total_train_loss:.4f}, Acc: {total_train_acc:.4f}")

        # 保存模型权重
        save_checkpoint(model, optimizer, epoch, cfg.save_path)

if __name__ == "__main__":
    train()
