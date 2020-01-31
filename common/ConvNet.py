from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义两个卷积层的厚度
depth = [4, 8]


class ConvNet(nn.Module):
    # 图像大小
    __image_size = 0
    # 分类类别
    __num_classes = 0

    def __init__(self, image_size, num_classes):
        super(ConvNet, self).__init__()
        self.__image_size = image_size
        self.__num_classes = num_classes

        # 定义第一层的卷积层
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        # 定义第一层的池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 定义第二层的卷积层
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)
        # 定义一个线性连接
        self.fc1 = nn.Linear(self.__image_size // 4 * self.__image_size // 4 * depth[1], 512)
        # 定义最后一个线性分类单元，输出为类别
        self.fc2 = nn.Linear(512, self.__num_classes)

    def forward(self, x):
        # 第一层卷积(batch_size, image_channels, image_width, image_height)
        x = self.conv1(x)
        # 使用ReLU激活函数，防止过拟合(batch_size, num_filters, image_width, image_height)
        x = F.relu(x)
        # 第二层池化(batch_size, depth[0], image_width/2, image_height/2)
        x = self.pool(x)
        # 第三层卷积(batch_size, depth[1], image_width/2, image_height/2)
        x = self.conv2(x)
        x = F.relu(x)
        # 第四层池化(batch_size, depth[1], image_width/4, image_height/4)
        x = self.pool(x)
        # 把数据变为一维数据(batch_size, depth[1] * image_width/4 * image_height/4)
        x = x.view(-1, self.__image_size // 4 * self.__image_size // 4 * depth[1])
        # 第五层为全链接层(batch_size, 512)
        x = F.relu(self.fc1(x))
        # 随即丢弃，防止过拟合
        x = F.dropout(x, training=self.training)
        # 第六层全链接层(batch_size, num_classes)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x



