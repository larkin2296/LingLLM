import torch
import torch.nn as nn
from model.minigpt import MiniLLM
from tokenizer import tokenize, VOCAB
from dataset import CharDataset
from torch.utils.data import DataLoader
from utils.read import load_data_from_dir
import csv

class Config:
    train_dir = "data/train/"
    val_dir = "data/val/"
    batch_size = 16
    max_seq_len = 64
    dropout = 0.1
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    epochs = 50
    save_path = "weights/minigpt_best.pth"

cfg = Config()


train_data = load_data_from_dir(cfg.train_dir)
val_data = load_data_from_dir(cfg.val_dir)
max_seq_len = cfg.max_seq_len
train_dataset = CharDataset(train_data, max_seq_len)
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,        # 或16，看显存/内存
    shuffle=True,        # 训练集打乱
    num_workers=2,       # mac上选2；Linux可更大
    drop_last=False,     # 保留全部样本
    # collate_fn=collate_fn,  # 如果你用变长/需要pad才加
)
val_dataset = CharDataset(val_data, max_seq_len)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

# 定义模型
# mps是苹果芯片专用加速
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
vocab_size = len(VOCAB)  # 词表总长
embed_dim = cfg.embed_dim        # embedding 的向量维度
num_heads = cfg.num_heads
model = MiniLLM(vocab_size, embed_dim, max_seq_len, num_heads, num_layers=cfg.num_layers, dropout=cfg.dropout).to(device)

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss(ignore_index=VOCAB.index("<pad>"))  # 忽略pad
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

best_val_loss = float('inf')
patience = 3            # 连续多少轮无提升则早停
counter = 0             # 记录未提升轮数
STOP_THRESHOLD = 0.0005 # val_loss低于多少可以直接early stop

log_file = "logs/train_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

# 训练主循环
def calc_accuracy(pred_logits, targets):
    # 预测概率最大的位置
    preds = pred_logits.argmax(dim=-1) # 取概率最大的下标
    mask = targets != VOCAB.index("<pad>") # 筛选不是pad的部分
    correct = (preds == targets) & mask # 取出预测正确的部分
    return correct.sum().item() / mask.sum().item() # 统计准确率

def train():
    epochs = cfg.epochs
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad() # 梯度清零
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1)) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            total_train_loss += loss.item()
            total_train_acc += calc_accuracy(logits.view(-1, vocab_size), y.view(-1))  # 统计准确率

        model.eval() # 切换到评估模式
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

        print(f"Epoch {epoch}: Train loss: {total_train_loss/len(train_loader):.4f}, "
            f"Train acc: {total_train_acc/len(train_loader):.4f} | "
            f"Val loss: {total_val_loss/len(val_loader):.4f}, "
            f"Val acc: {total_val_acc/len(val_loader):.4f}")
        
        # 早停策略
        # Early Stopping判断
        if total_val_loss < best_val_loss - 1e-5: # 有提升
            best_val_loss = total_val_loss
            counter = 0
            # 也可以这里保存最优模型
            torch.save(model.state_dict(), "minigpt_best.pth")
        else:
            counter += 1

        if total_val_loss < STOP_THRESHOLD or counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
        # 保存训练日志
        with open(log_file, "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, total_train_loss, total_train_acc, total_val_loss, total_val_acc])

    # 训练结束后保存权重
    torch.save(model.state_dict(), cfg.save_path)
    print("模型权重已保存到 minigpt.pth")

if __name__ == "__main__":
    print(f"训练数据条数: {len(train_data)}，验证数据条数: {len(val_data)}")
    train()