import os

# 要重命名的文件夹路径
folder_path = "./"

# 获取文件夹中的所有文件
files = os.listdir(folder_path)

# 遍历文件并重命名
for file_name in files:
    if file_name.startswith("objects_info_mask3d_"):
        new_name = file_name.replace("objects_info_mask3d_", "objects_info_mask3d_200c_")
        # 拼接完整的文件路径
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)
        # 执行重命名
        os.rename(old_path, new_path)
