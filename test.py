import argparse
parser = argparse.ArgumentParser(description="Description of your program")

# 添加--mode参数
parser.add_argument("--mode", type=str, choices=["eval", "result"], help="Mode of operation (eval or result)")
parser.add_argument("--times", type=str, nargs='+', help="List of times in 'yy-mm-dd-HH-MM-SS' format")
# 还可以添加其他参数
# parser.add_argument("--another_option", type=str, help="Description of another option")

# 解析命令行参数
args = parser.parse_args()

print("mode:",args.mode)
print("times:",args.times)