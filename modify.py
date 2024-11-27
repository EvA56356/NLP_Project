import json
import os

def process_json_file(file_path, output_file):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    with open(output_file, 'a') as out_f:
        for item in data:
            review_text = item.get("reviewText", "")
            overall = item.get("overall", 0)
            
            label = "good review" if overall > 3.5 else "bad review"
            
            new_entry = {"words": review_text, "label": label}
            
            out_f.write(json.dumps(new_entry) + "\n")


def process_all_files_in_directory(input_dir, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)
            print(f"Processing {file_name}...")
            process_json_file(file_path, output_file)
    print(f"All files processed. Results written to {output_file}")

input_dir = './'  
output_file = 'dev.txt'  
process_all_files_in_directory(input_dir, output_file)
