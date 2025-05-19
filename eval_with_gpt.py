from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
# 1. 读取验证集
VAL_DIR = "data/val/"
val_files = [os.path.join(VAL_DIR, f) for f in os.listdir(VAL_DIR) if f.endswith('.txt')]
eval_samples = []
for file in val_files:
    with open(file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                eval_samples.append((parts[0], parts[1]))  # (question, gold_answer)

print(f"验证集样本数：{len(eval_samples)}")

# 2. 调用你自己的小模型生成结果
# 假设你已经实现了如下函数：infer_one_sample(prompt: str) -> str
from infer_one import infer_one_sample  # 你得有这个函数，或者类似功能

# 3. 用ChatGPT自动打分/点评
client = OpenAI(api_key = os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
def gpt_score(question, model_reply, gold_reply=None):
    # 你可以自定义评分prompt
    sys_prompt = (
        "你是一个专业的中文写作评测AI，现在要对AI助手生成的写作回复质量打分。"
        "请严格从内容相关性、表达流畅性、用词准确性三个维度综合评分，1分（极差）-5分（极佳），并用一句话点评理由。"
        "评分格式示例：[分数] 理由"
    )
    user_prompt = f"用户问题：{question}\nAI答复：{model_reply}\n"
    if gold_reply:
        user_prompt += f"参考答案：{gold_reply}\n"
    user_prompt += "请给AI答复一个总分和简要评价："

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=128,
        stream=False
    )
    text = resp.choices[0].message.content.strip()
    # 解析分数
    import re
    match = re.match(r"[^\d]*(\d(\.\d)?)[^\d]*(.*)", text)
    if match:
        score = float(match.group(1))
        reason = match.group(3).strip()
        return score, reason, text
    else:
        return None, text, text

# 4. 自动批量评测
scores, comments = [], []
N = 20
for idx, (q, gold) in enumerate(eval_samples[:N]):
    reply = infer_one_sample(q)
    score, reason, raw = gpt_score(q, reply, gold)
    print(f"{idx+1}/{len(eval_samples)} | 分数: {score} | 题目: {q}\n模型答复: {reply}\n理由: {reason}\n")
    scores.append(score)
    comments.append((q, reply, score, reason, raw))

# 5. 计算平均分数，写入日志
valid_scores = [s for s in scores if s is not None]
avg_score = sum(valid_scores) / len(valid_scores)
print(f"\nChatGPT评测平均分：{avg_score:.2f}")

with open("logs/auto_eval_score.txt", "w") as f:
    f.write(str(avg_score))

with open("logs/eval_detail.csv", "w", encoding="utf-8") as f:
    f.write("问题,模型答复,分数,理由,原始回复\n")
    for q, reply, score, reason, raw in comments:
        row = [q.replace(",", "，"), reply.replace(",", "，"), str(score), reason.replace(",", "，"), raw.replace(",", "，")]
        f.write(",".join(row) + "\n")
