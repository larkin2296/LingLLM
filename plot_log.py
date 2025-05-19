import pandas as pd
import matplotlib.pyplot as plt

log_df = pd.read_csv("logs/train_log.csv")
plt.plot(log_df['epoch'], log_df['train_loss'], label='Train Loss')
plt.plot(log_df['epoch'], log_df['val_loss'], label='Val Loss')
plt.plot(log_df['epoch'], log_df['train_acc'], label='Train Acc')
plt.plot(log_df['epoch'], log_df['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.legend()
plt.title('MiniLLM 训练曲线')
plt.savefig("logs/curve.png")
plt.show()