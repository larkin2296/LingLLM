import os

def load_data_from_dir(data_dir, max_seq_len=128):
    """
    递归读取目录下所有txt文件，返回一个包含所有非空行的list
    """
    dialogs = []
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith('.txt'):
                fpath = os.path.join(root, fname)
                if os.path.getsize(fpath) > 0:
                    with open(fpath, encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                        # 根据 max_seq_len 截取文本
                        for i in range(0, len(text), max_seq_len):
                            dialogs.append(text[i:i+max_seq_len])
    return dialogs
