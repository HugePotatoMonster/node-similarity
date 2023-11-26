import datetime
import pickle
import os

def save_result(data, label):
    result_folder_path = os.path.join(os.getcwd(), "results", label)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{label}-{current_time}.pkl"
    with open(os.path.join(result_folder_path, filename), 'wb') as file:
        pickle.dump(data, file)

def read_result(pkl_path):
    try:
        with open(pkl_path, 'rb') as file:
            data = pickle.load(file)

        return data

    except FileNotFoundError:
        print(f"{pkl_path} 文件不存在")

    except Exception as e:
        print(f"读取文件时出现错误: {e}")