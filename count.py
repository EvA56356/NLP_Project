file_path = "convert_dataset/Toys_and_Games_5.json"  # 文件路径

# 打开文件并逐行计数
line_count = 0
with open(file_path, "r", encoding="utf-8") as file:
    for _ in file:
        line_count += 1

print(f"文件 {file_path} 的总行数为: {line_count}")
