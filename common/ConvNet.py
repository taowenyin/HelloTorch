import torch
import torch.nn as nn

# 定义两个卷积层的厚度
depth = [4, 8]


class ConvNet(nn.Module):
    def __init__(self, image_size, num_classes):
        super(ConvNet, self).__init__()
        # 图像大小
        self.image_size = image_size
        # 分类类别
        self.num_classes = num_classes

        # 定义第一层的卷积层
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        # 定义第一层的池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 定义第二层的卷积层
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)
        # 定义一个线性连接
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)
