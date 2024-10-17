import os

def rename_files_in_order(folder_path):
    files = os.listdir(folder_path)

    new_names = list(range(1, len(files) + 1))  # 新的文件名列表，从1开始计数

    for old_name, new_name in zip(files, new_names):
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, str(new_name) + ".txt")
        os.rename(old_path, new_path)

if __name__ == "__main__":
    folder_path = r"D:\pythonproject\yolov7-main\inference\MSRS-detection-label-master\MSRS-detection-label-master\infrared\test"  # 替换为你的文件夹路径

    rename_files_in_order(folder_path)
