import json

def check_invalid_data(file_path):
    invalid_samples = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            try:
                data = json.loads(line.strip()) 
                words = data.get("words", "").strip()
                label = data.get("label", "").strip()
                
                if not words: 
                    invalid_samples.append((line_num, "Empty 'words'", data))
                elif len(words.split()) == 0: 
                    invalid_samples.append((line_num, "'words' length is 0", data))
                elif not label:  
                    invalid_samples.append((line_num, "Empty 'label'", data))
            except json.JSONDecodeError as e:
                invalid_samples.append((line_num, f"JSON decode error: {e}", line.strip()))
    
    return invalid_samples


def main():
    file_path = "datasets/train.txt" 
    invalid_samples = check_invalid_data(file_path)
    
    if invalid_samples:
        print(f"Found {len(invalid_samples)} invalid samples:")
        for line_num, error, data in invalid_samples:
            print(f"Line {line_num}: {error} -> {data}")
    else:
        print("No invalid samples found.")

if __name__ == "__main__":
    main()
