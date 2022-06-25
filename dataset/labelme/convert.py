# coding=utf8

import os
import cv2


if __name__ == '__main__':
    # this_file_path = __file__
    # this_dir_path = os.path.dirname(this_file_path)

    this_dir_path = './'
    json_index = 1
    png_index = 1
    for file in os.listdir(this_dir_path):
        file_path = os.path.join(this_dir_path, file)
        ex = os.path.splitext(file_path)[-1]
        if ex != '.jpg' and ex != '.json' and ex != '.py':
            img = cv2.imread(file_path)
            new_file_path = file_path.replace(ex, '.jpg')
            cv2.imwrite(new_file_path, img)
            print(file_path+'---->'+new_file_path)
            os.remove(file_path)
