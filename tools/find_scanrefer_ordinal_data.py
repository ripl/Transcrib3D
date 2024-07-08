import json

# 打开原始 JSON 文件
with open('./data/scanrefer/ScanRefer_filtered_train_sampled1000.json', 'r') as file:
    data = json.load(file)

# 指定需要搜索的字符串列表
target_strings = ["from the left", "from the right", "in the middle"]

# 创建一个空列表，用于存储符合条件的数据
filtered_data = []

# 遍历原始数据，筛选出包含指定字符串的数据
for item in data:
    if 'description' in item and any(target in item['description'] for target in target_strings):
        filtered_data.append(item)

# 将符合条件的数据保存到新的 JSON 文件
with open('./data/scanrefer/ScanRefer_filtered_train_ordinal.json', 'a') as output_file:
    json.dump(filtered_data, output_file, indent=4)

print(f"找到并保存了{len(filtered_data)}个符合条件的数据到ScanRefer_filtered_train_ordinal.json文件。")
