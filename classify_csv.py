import csv

# 输入文件名和输出文件名
r_type='allocentric'
input_file = 'data/referit3d/sr3d_test.csv'
output_file = 'data/referit3d/sr3d_test_%s.csv'%r_type

# 打开输入和输出文件
with open(input_file, 'r', newline='') as input_csvfile, open(output_file, 'w', newline='') as output_csvfile:
    # 创建 CSV 读写对象
    csv_reader = csv.DictReader(input_csvfile)
    fieldnames = csv_reader.fieldnames
    csv_writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)

    # 写入输出文件的表头
    csv_writer.writeheader()

    # 遍历输入文件的每一行
    for row in csv_reader:
        # 检查 'coarse_reference_type' 列的值是否为 'horizontal'
        if row['coarse_reference_type'] == r_type:
            # 如果是 'horizontal'，则将行写入输出文件
            csv_writer.writerow(row)

print(f"'horizontal' 数据已提取并保存到 {output_file}")



