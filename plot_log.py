import os
import pandas as pd
import matplotlib.pyplot as plt
import csv

# 文件路径
log_file = "logs/train_log.csv"
history_file = "logs/train_history.csv"
curve_dir = "logs/curves"

# 确保曲线文件夹存在
os.makedirs(curve_dir, exist_ok=True)

def append_to_history():
    """
    将train_log.csv的数据追加到train_history.csv文件中
    """
    # 读取train_log.csv
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)

        # 如果train_history.csv文件不存在，创建并写入表头
        if not os.path.exists(history_file):
            with open(history_file, "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        # 将train_log.csv中的数据追加到train_history.csv
        log_df.to_csv(history_file, mode='a', header=False, index=False, encoding="utf-8")

        plot_training_curve()

        # 删除train_log.csv，确保下次训练时是独立的
        os.remove(log_file)

        print(f"数据已追加到 {history_file} 并删除了 {log_file}")
    else:
        print(f"错误: {log_file} 文件不存在！")

def plot_training_curve():
    """
    绘制并保存训练曲线图
    """
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        plt.plot(log_df['epoch'], log_df['train_loss'], label='Train Loss')
        plt.plot(log_df['epoch'], log_df['val_loss'], label='Val Loss')
        plt.plot(log_df['epoch'], log_df['train_acc'], label='Train Acc')
        plt.plot(log_df['epoch'], log_df['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.legend()
        plt.title('MiniLLM 训练曲线')

        # 保存曲线图，文件名可以加入epoch来区分
        plt.savefig(f"{curve_dir}/curve_epoch_{log_df['epoch'].iloc[-1]}.png")
        plt.close()

        print(f"训练曲线已保存为 {curve_dir}/curve_epoch_{log_df['epoch'].iloc[-1]}.png")
    else:
        print(f"错误: {log_file} 文件不存在！")

if __name__ == "__main__":
    # 执行数据保存并生成图像
    append_to_history()