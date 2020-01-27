import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

# 图像的尺寸
image_size = 28
# 标签的种类数
num_class = 10
# 训练的循环周期
num_epochs = 10
# 批次大小
batch_size = 64


if __name__ == '__main__':
    # 自动下载数据集，并提取训练数据，同时自动转化为Tensor数据
    train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor, download=True)
    # 获取测试数据
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor)
    # 加载训练数据，并对数据切分及打乱
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)