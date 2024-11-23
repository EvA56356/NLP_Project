import os
import json

# 定义输入和输出文件夹路径
input_folder = "new_dataset"
output_folder = "convert_dataset"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 定义批处理大小
batch_size = 1000  # 每批处理的 JSON 对象数量

# 分批处理函数
def process_large_file(input_path, output_path, batch_size):
    with open(input_path, "r") as file:
        json_array = []  # 用于存储当前批次的 JSON 对象
        batch_count = 0

        # 打开输出文件以分批写入
        with open(output_path, "w") as output:
            output.write("[")  # 写入数组的开头
            
            for line in file:
                try:
                    # 逐行读取并解析 JSON 对象
                    json_object = json.loads(line.strip())
                    json_array.append(json_object)

                    # 如果达到批量大小，写入文件并清空内存
                    if len(json_array) >= batch_size:
                        if batch_count > 0:
                            output.write(",")  # 添加逗号分隔
                        output.write(json.dumps(json_array, indent=4)[1:-1])  # 写入当前批次
                        json_array = []  # 清空内存中的批次数据
                        batch_count += 1

                except json.JSONDecodeError as e:
                    print(f"跳过无效的 JSON 行: {line.strip()}")

            # 写入最后一批
            if json_array:
                if batch_count > 0:
                    output.write(",")
                output.write(json.dumps(json_array, indent=4)[1:-1])

            output.write("]")  # 写入数组的结尾

    print(f"文件处理完成: {output_path}")

# 遍历输入文件夹中的所有 JSON 文件
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):  # 只处理 .json 文件
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)
        
        print(f"开始处理文件: {input_file_path}")
        process_large_file(input_file_path, output_file_path, batch_size)

print("所有文件已处理完成！")
