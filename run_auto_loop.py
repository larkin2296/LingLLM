import os
import subprocess

TARGET_SCORE = 4.6   # ChatGPT评测平均分
MAX_ROUNDS = 10

for round_idx in range(MAX_ROUNDS):
    print(f"\n======= Round {round_idx+1} Start =======")
    # Step 1. 正常训练
    subprocess.run(["python3", "train_main.py"])
    
    # Step 2. 自动评测（对验证集或专门test集）
    # 你的eval_with_gpt.py会把评测均分写到logs/auto_eval_score.txt
    subprocess.run(["python3", "eval_with_gpt.py"])
    with open("logs/auto_eval_score.txt") as f:
        avg_score = float(f.read().strip())
    print(f"[INFO] 当前模型ChatGPT自动评测均分：{avg_score}")
    
    if avg_score >= TARGET_SCORE:
        print("自动训练目标达标，流程终止。")
        break

    # Step 3. 自动增广（针对模型薄弱点）
    subprocess.run(["python3", "data_aug_chatgpt.py"])

    print(f"Round {round_idx+1} 完成，自动进入下一轮...")
