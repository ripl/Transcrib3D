import json
import random

# 读取原始JSON文件
with open('./data/scanrefer/ScanRefer_filtered_val.json', 'r') as f:
    original_data = json.load(f)

# 随机采样50个data
num_samples = 1000
random_samples = random.sample(original_data, num_samples)

# 保存为新的JSON文件
with open('./data/scanrefer/ScanRefer_filtered_val_sampled1000.json', 'w') as f:
    json.dump(random_samples, f, indent=4)

print("Sampled data saved to 'sampled.json'")
