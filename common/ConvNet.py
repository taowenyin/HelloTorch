from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义两个卷积层的厚度
depth = [4, 8]


# 卷积神经
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

    # image_size = image_size + 2 * padding - kernel_size + step
    # 实现前向运算
    def forward(self, *input: Any, **kwargs: Any):
        # 获取输入的数据(batch_size, image_channels, image_width, image_height)
        data = input[0]

        # 第一层卷积(batch_size, image_channels, image_size, image_size)
        data = self.conv1(data)
        # 使用ReLU激活函数，防止过拟合
        data = F.relu(data)
        # 第一层池化(batch_size, image_channels, image_size / pool_size, image_size / pool_size)
        data = self.pool(data)

        # 第二层卷积
        data = self.conv2(data)
        # 使用ReLU激活函数，防止过拟合
        data = F.relu(data)
        # 第二层池化
        data = self.pool(data)

        # 将立方体的特征图转化为一维向量
        data = data.view(-1, data.shape[1] * data.shape[2] * data.shape[3])

        # 第三层全链接层
        data = self.fc1(data)
        # 使用ReLU激活函数，防止过拟合
        data = F.relu(data)

        # 执行dropout，防止过拟合
        data = F.dropout(data, training=self.training)

        # 第四层全链接层，输出结果
        data = self.fc2(data)
        # 使用softmax激活函数
        data = F.log_softmax(data, dim=0)

        return data

    # 保存特征图
    def record_features(self, data):
        # 第一层卷积
        data = self.conv1(data)
        # 使用ReLU激活函数，防止过拟合
        feature_map1 = F.relu(data)
        # 第一层池化
        data = self.pool(feature_map1)

        # 第二层卷积
        data = self.conv2(data)
        # 使用ReLU激活函数，防止过拟合
        feature_map2 = F.relu(data)

        return feature_map1, feature_map2

