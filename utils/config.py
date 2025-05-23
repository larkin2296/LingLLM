class Config:
    # 训练集地址
    train_bin = "data/train/train.bin"
    batch_size = 2
    # 最大序列长度
    max_seq_len = 512
    # 它能在训练时随机“丢弃”一部分神经元（设为0），防止模型对训练数据“死记硬背”而失去泛化能力。
    dropout = 0.2
    # 模型维度
    embed_dim = 128
    # 多头注意力机制的头数
    num_heads = 4
    # Transformer Block的层数
    num_layers = 4
    # 训练轮数
    epochs = 80
    # 模型保存路径
    save_path = "weights/sft_minigpt_best.pth"
    checkpoint_path = "weights/checkpoint.pth"
    pre_save_path = "weights/minigpt_best.pth"
    learning_rate = 1e-4
    pre_log_file = "logs/pre_train_log.csv"
    sft_log_file = "logs/sft_train_log.csv"