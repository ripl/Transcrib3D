import csv

# 定义原始CSV文件和新CSV文件的文件名
input_csv_file = "data/referit3d/nr3d_train_sampled1000.csv"
output_csv_file = "data/referit3d/nr3d_train_sampled1000_lr.csv"

# 打开原始CSV文件进行读取，并指定UTF-8编码
with open(input_csv_file, mode='r', encoding='utf-8') as csv_file:
    # 创建CSV读取器
    csv_reader = csv.DictReader(csv_file)
    
    # 提取满足条件的行并保存到一个列表中
    filtered_rows = [row for row in csv_reader if any(keyword in row['utterance'] for keyword in ['left', 'right', 'leftmost', 'rightmost'])]

# 将满足条件的行写入新的CSV文件
with open(output_csv_file, mode='w', encoding='utf-8', newline='') as csv_file:
    # 创建CSV写入器
    fieldnames = filtered_rows[0].keys()
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # 写入CSV文件的表头
    csv_writer.writeheader()
    
    # 写入满足条件的行
    csv_writer.writerows(filtered_rows)

print("提取并保存完成。")
