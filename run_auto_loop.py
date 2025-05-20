import os
import subprocess
import sys

TARGET_SCORE = 4.6   # ChatGPT评测平均分
MAX_ROUNDS = 100

ALLOW_EARLY_STOP = False

for round_idx in range(MAX_ROUNDS):
    print(f"\n======= Round {round_idx+1} Start =======")
    # Step 1. 正常训练
    if round_idx < 3:
        # 前3个epoch不使用早停
        ALLOW_EARLY_STOP = False
    else:
        ALLOW_EARLY_STOP = True
    subprocess.run([
        sys.executable,
        "train_main.py",
        "--allow_early_stop",
        str(int(ALLOW_EARLY_STOP))
    ], check=True)
    
    # Step 2. 自动评测（对验证集或专门test集）
    # 你的eval_with_gpt.py会把评测均分写到logs/auto_eval_score.txt
    subprocess.run([
        sys.executable,
        "eval_with_gpt.py"
    ])
    with open("logs/auto_eval_score.txt") as f:
        avg_score = float(f.read().strip())
    print(f"[INFO] 当前模型ChatGPT自动评测均分：{avg_score}")
    
    if avg_score >= TARGET_SCORE:
        print("自动训练目标达标，流程终止。")
        break

    # Step 3. 自动增广（针对模型薄弱点）
    subprocess.run([sys.executable, "data_aug_chatgpt.py"])

    print(f"Round {round_idx+1} 完成，自动进入下一轮...")
