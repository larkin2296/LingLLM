from dotenv import load_dotenv
from openai import OpenAI
import os
import re

load_dotenv()  # 加载环境变量
client = OpenAI(api_key = os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

def generate_samples(topic="交流", num_samples=20):
    # 可以丰富模板/问题
    prompts = [
        f"请写一组与{topic}能力提升相关的问答，每组包含用户的问题和AI助手的专业、友好答复，输出30组，格式严格如下：Q: xxx\nA: xxx",
        # 你可以自定义多种风格/体裁问题
    ]
    all_samples = []
    for prompt in prompts:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是沟通交流能力提升的专家AI，善于与人交流。"},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=2048,
            n=1,
        )
        # 解析输出
        result = response.choices[0].message.content
        samples = [line for line in result.split('\n') if line.strip()]
        # 拆分Q/A
        qa_pairs = []
        for i in range(0, len(samples), 2):
            if i + 1 < len(samples):
                q = samples[i].replace("Q: ", "").strip()
                a = samples[i+1].replace("A: ", "").strip()
                qa_pairs.append((q, a))
        all_samples.extend(qa_pairs)
    return all_samples

def save_samples_to_txt(samples, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for q, a in samples:
            f.write(f"{q}\t{a}\n")  # 用tab分隔，后续tokenize时拆分

def get_next_train_filename(dir_path, prefix="auto_", ext=".txt"):
    # 找出目录下所有 auto_数字.txt 文件
    pattern = re.compile(rf"{prefix}(\d+){ext}$")
    nums = []
    for fname in os.listdir(dir_path):
        match = pattern.match(fname)
        if match:
            nums.append(int(match.group(1)))
    next_num = max(nums, default=0) + 1
    return os.path.join(dir_path, f"{prefix}{next_num:03d}{ext}")

if __name__ == "__main__":
    samples = generate_samples(topic="交流", num_samples=40)
    save_path = get_next_train_filename("data/train/")
    save_samples_to_txt(samples, save_path)
    print(f"生成样本数：{len(samples)} | 保存文件：{save_path}")
