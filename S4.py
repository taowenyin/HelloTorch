import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from common.ConvNet import ConvNet

# 图像的尺寸
image_size = 28
# 标签的种类数
num_class = 10
# 训练的循环周期
num_epochs = 10
# 批次大小
batch_size = 64

def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素


if __name__ == '__main__':
    # 自动下载数据集，并提取训练数据，同时自动转化为Tensor数据
    train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    # 获取测试数据
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    # 加载训练数据，并对数据切分及打乱
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    indices = range(len(test_dataset))
    # 生成验证集索引
    indices_val = indices[:5000]
    # 生成测试集索引
    indices_test = indices[5000:]
    sample_val = SubsetRandomSampler(indices_val)
    sample_test = SubsetRandomSampler(indices_test)
    # 定义验证集和测试集的加载器
    validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                   shuffle=False,  sampler=sample_val)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=False,  sampler=sample_test)

    # 构建网络对象
    net = ConvNet(image_size, num_class)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 记录准确率等数值
    record = []
    # 记录卷积核
    weights = []

    for epoch in range(num_epochs):
        # 记录训练数据集准确率的容器
        train_rights = []

        for batch_idx, (data, target) in enumerate(train_loader):
            net.train()
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            optimizer.step()
            right = rightness(output, target)
            train_rights.append(right)

            if batch_idx % 100 == 0:
                net.eval()
                val_rights = []

                # 验证集
                for(data, target) in validation_loader:
                    output = net(data)
                    right = rightness(output, target)
                    val_rights.append(right)