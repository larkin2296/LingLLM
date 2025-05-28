import os
from utils.oss_upload import upload_file_to_oss, download_file_from_oss

def list_project_files(base_path='.'):
    # 排除这些目录
    exclude_dirs = {'venv', '.git'}
    files = []
    for root, dirs, filenames in os.walk(base_path):
        # 修改dirs以原地过滤（影响os.walk递归）
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for filename in filenames:
            filepath = os.path.relpath(os.path.join(root, filename), base_path)
            files.append(filepath)
    return files

def main():
    print("请选择操作：")
    print("1. 上传文件到 OSS")
    print("2. 从 OSS 下载文件到本地")
    choice = input("输入 1 或 2：").strip()

    if choice == '1':
        files = list_project_files('.')
        print("\n项目内可选文件：")
        for idx, f in enumerate(files):
            print(f"{idx+1}: {f}")
        file_idx = int(input("输入要上传的文件编号：")) - 1
        local_file = files[file_idx]
        oss_file = input("请输入上传到 OSS 的路径（例如 myfolder/filename.txt）：").strip()
        upload_file_to_oss(oss_file, local_file)
        print(f"已上传 {local_file} 到 OSS 路径 {oss_file}")

    elif choice == '2':
        oss_file = input("请输入要下载的 OSS 文件路径（例如 myfolder/filename.txt）：").strip()
        local_file = input("请输入下载到本地的文件路径（例如 data/filename.txt）：").strip()
        download_file_from_oss(oss_file, local_file)
        print(f"已下载 OSS 文件 {oss_file} 到本地 {local_file}")

    else:
        print("无效的选择，请重新运行。")

if __name__ == '__main__':
    main()
