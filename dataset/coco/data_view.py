import cv2
import os
import numpy as np

from pycocotools.coco import COCO


def draw_rectangle(coordinates, image, image_name):
    for coordinate in coordinates:
        left, top, right, bottom, label = map(int, coordinate)
        color = colors[label % len(colors)]
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(image, str(label), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    cv2.imwrite(save_path + '/' + image_name, image)

if __name__ == '__main__':
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    img_path = '/home/taowenyin/MyCode/Dataset/fire_coco/val2017'
    annFile = '/home/taowenyin/MyCode/Dataset/fire_coco/annotations/instances_val2017.json'
    save_path = '/home/taowenyin/MyCode/Dataset/fire_coco/output/val'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    coco = COCO(annFile)

    imgIds = coco.getImgIds()

    for imgId in imgIds:

        img = coco.loadImgs(imgId)[0]
        image_name = img['file_name']
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
        anns = coco.loadAnns(annIds)

        coordinates = []
        img_raw = cv2.imread(os.path.join(img_path, image_name))
        for j in range(len(anns)):
            coordinate = anns[j]['bbox']
            coordinate[2] += coordinate[0]
            coordinate[3] += coordinate[1]
            coordinate.append(anns[j]['category_id'])
            coordinates.append(coordinate)

        draw_rectangle(coordinates, img_raw, image_name)

