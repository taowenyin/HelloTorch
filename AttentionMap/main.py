import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.models import mobilenet_v2, vgg16
from PIL import Image


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def visualize_grid_attention_map(img_file, attention_mask=None, ratio=1, cmap="jet",
                                 is_save_result=False, save_path=None):
    """
    注意力机制的热力图可视化

    Parameters
    ----------
    img_file: 载入图像的路径
    attention_mask: 注意力机制的掩码
    ratio: 输出图像的缩放比例
    cmap: 注意力机制热力图的风格，默认为jet
    is_save_result: 是否保存结果
    save_path: 保存热力图可视化的路径
    """

    img_file = os.path.join(os.path.dirname(__file__), img_file)

    # 读入图像
    image = Image.open(os.path.join(img_file), mode='r')
    image_h, image_w = int(image.size[0] * ratio), int(image.size[1] * ratio)

    plt.subplot(1, 1, 1)
    image = image.resize((image_h, image_w))
    plt.imshow(image, alpha=1)

    mask = cv2.resize(attention_mask, (image_h, image_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    # 添加Color Bar
    plt.colorbar()

    if is_save_result:
        save_path = os.path.join(os.path.dirname(__file__), save_path)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        img_name = img_file.split('/')[-1].split(".")[0] + '_with_attention.jpg'
        img_with_attention_save_path = os.path.join(save_path, img_name)

        plt.savefig(img_with_attention_save_path, dpi=100)

    plt.show()


def input_transform(resize=(64, 64)):
    """
    对图像进行转换

    :param resize: 转换后的图像大小(H, W)
    :return: 返回转换对象
    """

    if resize[0] > 0 and resize[1] > 0:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


if __name__ == '__main__':
    cuda = False
    if cuda and not torch.cuda.is_available():
        raise Exception("没有找到GPU，运行时添加参数 --no_cuda")
    device = torch.device("cuda" if cuda else "cpu")

    net = mobilenet_v2(pretrained=True)
    encoding_model = nn.Sequential(*list(net.features.children()))
    encoding_model.to(device)

    encoding_model.eval()

    img_file = os.path.join(os.path.dirname(__file__), 'img/example.jpg')
    image = input_transform()(Image.open(img_file))
    # 增加B的维度
    image = image.unsqueeze(0)

    feature_map = encoding_model(image)

    print('xx')

    # attention_mask = np.random.randn(64)
    # normed_attention_mask = softmax(attention_mask).reshape(8, 8)
    # visualize_grid_attention_map(img_file='img/example.jpg',
    #                              attention_mask=normed_attention_mask,
    #                              ratio=1,
    #                              cmap='jet',
    #                              is_save_result=True,
    #                              save_path='img')
    #
    # print(net)
    #
    # net = vgg16(pretrained=True)
    # print('========================================')
    # print(net)
