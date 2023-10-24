# encoding:utf-8
import csv
import random

# 原始CSV文件路径
input_csv_path = 'data/referit3d/sr3d_test.csv'  # 请替换为您的原始CSV文件路径

# 新的CSV文件路径
output_csv_path = 'data/referit3d/sr3d_test_sampled1000.csv'  # 请替换为您想要保存的新CSV文件路径

# 采样的数据行数
sample_size = 1000

# 从原始文件读取数据
# 从原始文件读取数据
with open(input_csv_path, 'r', encoding='utf-8') as input_file:
    csv_reader = csv.reader(input_file)
    header = next(csv_reader)  # 读取头部


    # 从剩余的数据中随机采样1000条
    sampled_data = random.sample(list(csv_reader), sample_size)

# 将采样的数据写入新的CSV文件
with open(output_csv_path, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(header)  # 写入头部
    csv_writer.writerows(sampled_data)

print(f'已从原始文件中随机采样{sample_size}条数据，并保存到新的CSV文件：{output_csv_path}')
