import json
import os
import shutil

from tqdm import tqdm


def clear_repeat_file(dataset_path, data_folder):
    temp_files = os.listdir(dataset_path)
    total_files = []
    for file in tqdm(temp_files):
        if file.endswith('.json'):
            total_files.append(file)
            file_full_path = os.path.join(dataset_path, file)
            with open(file_full_path, "r") as f:
                raw_data = json.load(f)
                imagePath = raw_data['imagePath']
                image_full_path = os.path.join(dataset_path, imagePath)

                shutil.copy(file_full_path, data_folder)
                shutil.copy(image_full_path, data_folder)


if __name__ == '__main__':
    # 数据集根路径
    dataset_path = '/home/taowenyin/MyCode/Dataset/fire/instance_segmentation/fire'

    data_folder = os.path.join(dataset_path, 'data')

    if os.path.exists(data_folder):
        os.removedirs(data_folder)

    os.makedirs(data_folder)
    clear_repeat_file(dataset_path, data_folder)

    print('')