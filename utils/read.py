import os

def load_data_from_dir(data_dir):
    """
    递归读取目录下所有txt文件，返回一个包含所有非空行的list
    """
    dialogs = []
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith('.txt'):
                fpath = os.path.join(root, fname)
                with open(fpath, encoding='utf-8') as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            dialogs.append(s)
    return dialogs
