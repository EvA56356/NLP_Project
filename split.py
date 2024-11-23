import json
import os
from tqdm import tqdm  # 使用 tqdm 库显示进度条

def split_json_file(file_path, output_dir):
    # 读取 JSON 文件
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 确定总行数
    total_lines = len(data)
    max_lines = min(total_lines, 1000000)
    
    # 确定划分数量
    split_80 = max_lines * 80 // 100
    split_10_validation = max_lines * 10 // 100
    split_10_training = max_lines - split_80 - split_10_validation

    # 数据分割
    test_data = data[:split_80]
    validation_data = data[split_80:split_80 + split_10_validation]
    training_data = data[split_80 + split_10_validation:split_80 + split_10_validation + split_10_training]
    
    # 创建输出目录
    os.makedirs(f"{output_dir}/test", exist_ok=True)
    os.makedirs(f"{output_dir}/validation", exist_ok=True)
    os.makedirs(f"{output_dir}/training", exist_ok=True)
    
    # 获取文件名
    filename = os.path.basename(file_path)
    
    # 保存到文件，显示保存进度
    print(f"Processing {filename} with {total_lines} total lines (processing up to {max_lines} lines)...")
    with tqdm(total=100, desc=f"Saving {filename}") as pbar:
        with open(f"{output_dir}/test/{filename}", 'w') as f:
            json.dump(test_data, f, indent=2)
        pbar.update(33)
        with open(f"{output_dir}/validation/{filename}", 'w') as f:
            json.dump(validation_data, f, indent=2)
        pbar.update(33)
        with open(f"{output_dir}/training/{filename}", 'w') as f:
            json.dump(training_data, f, indent=2)
        pbar.update(34)

    print(f"Processed {filename}: Total lines: {total_lines}, Processed lines: {max_lines}")
    print(f"Saved splits to {output_dir}")


def process_all_files_in_directory(input_dir, output_dir):
    # 遍历输入目录下的所有 JSON 文件
    json_files = [file_name for file_name in os.listdir(input_dir) if file_name.endswith('.json')]
    total_files = len(json_files)

    print(f"Found {total_files} JSON files in {input_dir}. Starting processing...")
    for i, file_name in enumerate(json_files, 1):
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing file {i}/{total_files}: {file_name}")
        split_json_file(file_path, output_dir)
        print(f"Finished processing file {i}/{total_files}: {file_name}\n")


# 使用示例
input_dir = 'convert_dataset'  # 替换为你的输入文件夹路径
output_dir = 'datasets/new_datasets'  # 替换为你的输出路径
process_all_files_in_directory(input_dir, output_dir)
