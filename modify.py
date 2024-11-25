import json
import os

def process_json_file(file_path, output_file):
    with open(file_path, 'r') as f:
        # 读取 JSON 数组
        data = json.load(f)
    
    # 打开输出文件以追加模式写入
    with open(output_file, 'a') as out_f:
        for item in data:
            # 提取 reviewText 和 overall
            review_text = item.get("reviewText", "")
            overall = item.get("overall", 0)
            
            # 判断 label
            label = "good review" if overall > 3.5 else "bad review"
            
            # 构造新的 JSON 格式
            new_entry = {"words": review_text, "label": label}
            
            # 写入文件，每条记录一行
            out_f.write(json.dumps(new_entry) + "\n")


def process_all_files_in_directory(input_dir, output_file):
    # 如果输出文件已存在，清空它
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # 遍历目录下的所有 JSON 文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):  # 只处理 JSON 文件
            file_path = os.path.join(input_dir, file_name)
            print(f"Processing {file_name}...")
            process_json_file(file_path, output_file)
    print(f"All files processed. Results written to {output_file}")


# 使用示例
input_dir = './'  # 替换为你的输入文件夹路径
output_file = 'dev.txt'  # 指定输出文件路径
process_all_files_in_directory(input_dir, output_file)
