"""
需要修改的地方
1. category_set = ['TBC']此处的类别信息需要修改
2. 代码末尾处的voc标注文件夹
3. 带末尾处对应生成的json文件名
注：训练集与测试集中类别的顺序必须保持一致，因此最好事先确定category的顺序，书写在category_set中

参考（本代码参考版本1）：
COCO 数据集解释：https://blog.csdn.net/guoqingru0311/article/details/121855345
VOC2COCO版本1：https://www.bilibili.com/video/av510107250
https://blog.csdn.net/effort_study/article/details/123763951
VOC2COCO版本2：https://github.com/jiachen0212/voc2coco-pattern
"""

import xml.etree.ElementTree as ET
import os
import json
import collections
import argparse

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

# category_set = dict()
image_set = set()
image_id = 1  # train:2018xxx; val:2019xxx; test:2020xxx
category_item_id = 1
annotation_id = 1
category_set = ['fire']

def addCatItem(name):
    '''
    增加json格式中的categories部分
    '''
    global category_item_id
    category_item = collections.OrderedDict()
    # 添加主类名
    # category_item['supercategory'] = 'none'
    category_item['supercategory'] = name
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_item_id += 1


def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    # image_item = dict()    #按照一定的顺序，这里采用collections.OrderedDict()
    image_item = collections.OrderedDict()
    # 使用原始的文件名
    jpg_name = os.path.splitext(file_name)[0] + '.jpg'
    # jpg_name = file_name
    image_item['file_name'] = jpg_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    image_item['id'] = image_id
    coco['images'].append(image_item)
    image_set.add(jpg_name)
    image_id = image_id + 1
    return image_id


def txt2list(txtfile):
    f = open(txtfile)
    l = []
    for line in f:
        l.append(line[:-1])
    return l


def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    # annotation_item = dict()
    annotation_item = collections.OrderedDict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])
    annotation_item['segmentation'].append(seg)
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_item['id'] = annotation_id
    annotation_id += 1
    coco['annotations'].append(annotation_item)


def parseXmlFiles(annotations_path, image_sets_file=None):
    if image_sets_file is None:
        annotations_list = os.listdir(annotations_path)
        annotations_list.sort()
    else:
        annotations_list = []
        for ind, xml_name in enumerate(txt2list(image_sets_file)):
            annotations_list.append(xml_name + '.xml')

    for f in annotations_list:
        if not f.endswith('.xml'):
            continue

        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(annotations_path, f)
        print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()  # 抓根结点元素

        if root.tag != 'annotation':  # 根节点标签
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            # elem.tag, elem.attrib，elem.text
            if elem.tag == 'folder':
                continue

            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')

            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)  # 图片信息
                    print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name))
                    # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None

                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    # if object_name not in category_set:
                    #    current_category_id = addCatItem(object_name)
                    # else:
                    # current_category_id = category_set[object_name]
                    current_category_id = category_set.index(object_name) + 1  # index默认从0开始,但是json文件是从1开始，所以+1
                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['xmin'])
                    # y
                    bbox.append(bndbox['ymin'])
                    # w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    # h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    print(
                        'add annotation with {},{},{},{}'.format(object_name, current_image_id - 1, current_category_id,
                                                                 bbox))
                    addAnnoItem(object_name, current_image_id - 1, current_category_id, bbox)
    # categories部分
    for categoryname in category_set:
        addCatItem(categoryname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VOC Convert COCO', add_help=False)
    parser.add_argument('--annotations_path', type=str, required=True)
    parser.add_argument('--image_sets_file', type=str)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    annotations_path = args.annotations_path
    image_sets_file = args.image_sets_file
    output_file = args.output_file
    parseXmlFiles(annotations_path, image_sets_file)
    json.dump(coco, open(output_file, 'w'))
