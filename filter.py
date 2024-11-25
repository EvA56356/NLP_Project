import json

def check_invalid_data(file_path):
    """
    检查数据文件中的无效样本并返回列表。
    :param file_path: 数据文件路径（假设每行一个 JSON 样本）。
    :return: 无效样本列表。
    """
    invalid_samples = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            try:
                data = json.loads(line.strip())  # 解析每一行 JSON
                words = data.get("words", "").strip()
                label = data.get("label", "").strip()
                
                # 检查无效条件
                if not words:  # 如果 'words' 字段为空
                    invalid_samples.append((line_num, "Empty 'words'", data))
                elif len(words.split()) == 0:  # 如果 'words' 没有有效单词
                    invalid_samples.append((line_num, "'words' length is 0", data))
                elif not label:  # 如果 'label' 字段为空
                    invalid_samples.append((line_num, "Empty 'label'", data))
            except json.JSONDecodeError as e:
                invalid_samples.append((line_num, f"JSON decode error: {e}", line.strip()))
    
    return invalid_samples


def main():
    file_path = "datasets/train.txt"  # 替换为你的数据文件路径
    invalid_samples = check_invalid_data(file_path)
    
    if invalid_samples:
        print(f"Found {len(invalid_samples)} invalid samples:")
        for line_num, error, data in invalid_samples:
            print(f"Line {line_num}: {error} -> {data}")
    else:
        print("No invalid samples found.")

if __name__ == "__main__":
    main()
