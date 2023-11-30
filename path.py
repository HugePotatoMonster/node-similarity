import datetime
import pickle
import os
import json

def save_result(data, label):
    result_root = os.path.join(os.getcwd(), "results")
    if not os.path.exists(result_root):
        os.makedirs(result_root)

    result_folder_path = os.path.join(result_root, label)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{label}-{current_time}.json"
    with open(os.path.join(result_folder_path, filename), 'w') as file:
        json.dump(data, file)

def load_result(pkl_path):
    try:
        with open(pkl_path, 'r') as file:
            data = json.load(file)

        return data

    except FileNotFoundError:
        print(f"{pkl_path} not found")

    except Exception as e:
        print(f"Error when reading file: {e}")
    
def save_ns(data, json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file)

def load_ns(json_file_path):
    try:
        with open(json_file_path, 'r') as json_file:
            s_ns_map = json.load(json_file)
        print(f"Loaded s_ns_map from {json_file_path}")
        return s_ns_map
    except FileNotFoundError:
        print(f"{json_file_path} not found. Creating a new one.")
        return {}