import torch
import torch.nn as nn
from model.minigpt import MiniLLM
from tokenizer import tokenize, VOCAB

# 假设只训练"hello ai"这个句子
sentence = "hello ai"
token_ids = tokenize(sentence)
vocab_size = len(VOCAB)  # 其实应为词表总长，但这里只做演示

# 设备设置
# 如果有GPU可用，使用GPU，否则使用CPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将输入数据转移到设备上
#input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)

# 转成PyTorch张量
input_ids = torch.tensor([token_ids], dtype=torch.long, requires_grad=True)  # batch=1

# 定义模型
embed_dim = 16 # embedding 的向量维度
max_seq_len = 32 # 句子最大长度
num_heads = 2
model = MiniLLM(vocab_size, embed_dim, max_seq_len, num_heads)

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
# 把模型里所有要学习的参数都传进来
# 学习率（learning rate），神经网络最关键的超参数之一,决定每次参数调整步子的大小
# weight_decay=1e-5 正则化（防止过拟合）
# betas（针对Adam的自适应参数）
# amsgrad（是否用AMSGrad变种）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5) # 根据梯度调整每个参数的数值，逐步让 loss 降低

# 训练一次示例
model.train()
for epoch in range(100):
    logits = model(input_ids)
    # 预测的目标是下一个token，所以输入和目标错位一位
    pred = logits[:, :-1, :].reshape(-1, vocab_size)
    pred = logits[:, :-1, :].reshape(-1, vocab_size)
    # 作用： 取模型预测结果（logits），除了最后一个token都要，用来预测“下一个token”。
    # logits: 形状 [batch, seq_len, vocab_size]，表示每个位置每个token的概率分布。
    # [:, :-1, :] 表示：所有batch，除了最后一个位置（假设句子有5个字，就是取0-3共4个，留最后一个给label用）。
    # reshape(-1, vocab_size) 把所有位置摊平，方便和 target 一一对应。
    # 最终效果： 每个预测“当前token应该输出哪个词”的分布。
    target = input_ids[:, 1:].reshape(-1)
    # 真实标签，是“输入序列的下一个token”，用作训练目标。
    loss = loss_fn(pred, target)
    # 计算损失
    optimizer.zero_grad()
    # 把模型中所有参数的梯度清零
    loss.backward()
    # 反向传播 自动计算梯度
    optimizer.step()
    # 更新参数
    if epoch % 10 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}")
    # 每训练10轮，打印一次当前轮数和损失值
    torch.save(model.state_dict(), "minigpt.pth")
    print("模型权重已保存到 minigpt.pth")