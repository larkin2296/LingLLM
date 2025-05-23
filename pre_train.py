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
import time
import platform

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

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and platform.system() == "Darwin":
    device = torch.device("mps")
else:
    device = torch.device("cpu")
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
# optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)

# ReduceLROnPlateau（自适应学习率衰减）
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5) # 如果loss（或val_loss/val_acc等监控指标）在若干epoch里没明显提升，就自动降低学习率。

# Cosine Annealing（余弦退火）
# 让学习率像余弦曲线一样“周期性减小”，一开始大、后面慢慢变小，甚至能多次反复。
# from torch.optim.lr_scheduler import CosineAnnealingLR
# scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Warmup（热身策略）
# 前10%（比如前10个epoch）线性增加学习率，从很小到你设的最大值
# from transformers import get_linear_schedule_with_warmup
# scheduler = get_linear_schedule_with_warmup(
#     optimizer, num_warmup_steps=100, num_training_steps=1000
# )

best_val_loss = float('inf')
patience = 3
counter = 0
STOP_THRESHOLD = 0.0005

if not os.path.exists(cfg.pre_log_file):
    with open(cfg.pre_log_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

# 恢复训练
def load_checkpoint(model, optimizer, filepath):
    if not os.path.exists(filepath):
        download_file_from_oss(filepath, filepath)
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
    global best_val_loss, counter
    # 设定你的目标
    TARGET_LOSS = 0.05    # 你希望train_loss小于0.05就停止
    TARGET_ACC = 0.95     # 你希望train_acc大于0.95就停止
    MIN_LOSS_DECREASE = 0.001  # 新增：最小下降幅度
    accum_steps = 16  # 梯度累积步数
    last_loss = None
    start_epoch = 0
    if os.path.exists(cfg.checkpoint_path):
        print("加载检查点...")
        start_epoch = load_checkpoint(model, optimizer, cfg.checkpoint_path)
        print(f"恢复训练从第 {start_epoch} 轮开始。")
    elif os.path.exists(cfg.pre_save_path):
        print("加载模型参数权重（无optimizer）...")
        if not os.path.exists(cfg.pre_save_path):
            download_file_from_oss(cfg.pre_save_path, cfg.pre_save_path)
        state = torch.load(cfg.pre_save_path, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)  # 兼容只存了权重的场景
        print("模型参数已加载，optimizer和epoch将从头初始化")
        # 这里start_epoch保持为0
    else:
        print("未找到权重，模型从头训练")
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start_time = time.time()  # 记录epoch开始时间

        model.train()
        total_train_loss = 0
        total_train_acc = 0
        count = 0  # 用于统计真实batch数

        optimizer.zero_grad()  # 梯度清零放到外面
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss = loss / accum_steps  # 累积步数归一化
            loss.backward()
            
            # 累积到指定步数才进行optimizer.step
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            # 累计loss/acc，注意只统计一次，不用除以accum_steps
            total_train_loss += loss.item() * accum_steps  # 还原为正常loss
            total_train_acc += calc_accuracy(logits.view(-1, vocab_size), y.view(-1))
            count += 1

        total_train_loss /= count
        total_train_acc /= count

        scheduler.step(total_train_loss)

        epoch_seconds = time.time() - epoch_start_time

        print(f"Pre-training Epoch {epoch}: Loss: {total_train_loss:.4f}, Acc: {total_train_acc:.4f}| Time: {epoch_seconds:.2f}S")

        # 保存模型权重
        save_checkpoint(model, optimizer, epoch, cfg.checkpoint_path)

        with open(cfg.pre_log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{total_train_loss},{total_train_acc}\n")

        if total_train_loss < TARGET_LOSS and total_train_acc > TARGET_ACC:
            print(f"达到目标！Loss: {total_train_loss:.4f}, Acc: {total_train_acc:.4f}，提前终止并保存权重。")
            break

        if last_loss is not None:
            delta_loss = abs(last_loss - total_train_loss)
            print(f"Loss下降: {delta_loss:.6f}")
            if delta_loss < MIN_LOSS_DECREASE:
                print(f"Loss下降幅度({delta_loss:.6f})小于{MIN_LOSS_DECREASE}，提前终止训练。")
                break
        last_loss = total_train_loss
    
    print(f"训练结束，权重已保存到 {cfg.pre_save_path}")
    torch.save(model.state_dict(), cfg.pre_save_path)
    # 训练结束后上传权重
    upload_file_to_oss(cfg.pre_save_path, cfg.pre_save_path)

if __name__ == "__main__":
    train()
