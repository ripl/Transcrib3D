import csv
import random

# 输入文件名列表
input_files = ['input1.csv', 'input2.csv', 'input3.csv', 'input4.csv', 'input5.csv']

# 输出文件名
output_file = 'data/referit3d/sr3d_test_assembled30x5.csv'

# 用于存储采样行的列表
sampled_rows = []

# 从每个输入文件中随机采样30行
for r_type in ['horizontal','vertical','between','support','allocentric']:
    input_file='data/referit3d/sr3d_test_%s.csv'%r_type
    with open(input_file, 'r', newline='') as input_csvfile:
        csv_reader = csv.DictReader(input_csvfile)
        rows = list(csv_reader)
        sampled_rows.extend(random.sample(rows, 30))

# 将采样的行写入输出文件
with open(output_file, 'w', newline='') as output_csvfile:
    fieldnames = sampled_rows[0].keys()
    csv_writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(sampled_rows)

print(f"采样数据已保存到 {output_file}")
