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


# 计算预测的损失函数
def rightness(predictions, labels):
    # 计算每批中每个元素概率最大的索引
    pred = torch.max(predictions.data, 1)[1]
    # 将预测的下标标签与实际标签进行比较，并求和，得到正确的数量
    rights = pred.eq(labels.data.view_as(pred)).sum()
    # 返回正确的数量和这一次一共比较了多少元素
    return rights, len(labels)


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

    # 数据读取测试
    # idx = 100
    # img = train_dataset[idx]
    # img_data = img[0][0].numpy()
    # img_label = img[1]
    #
    # plt.imshow(img_data)
    # plt.show()
    # print('标签是：', img_label)

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
            data, target = data.clone(), target.clone().detach()

            # 清空梯度
            optimizer.zero_grad()
            # 打开dropout标记
            net.train()
            # 完成一次前馈计算，并得到预测值
            output = net(data)
            # 计算损失函数
            loss = criterion(output, target)
            # 反向传播
            loss.backward()
            # 梯度下降法
            optimizer.step()

            right = rightness(output, target)
            # 记录预测损失函数结果
            train_rights.append(right)

            # 每执行100个批次打印一次结果
            if batch_idx % 100 == 0:
                # 关闭dropout标记
                net.eval()
                # 记录在校验集上的准确度
                val_rights = []

                # 验证集
                for(data, target) in validation_loader:
                    data, target = data.clone(), target.clone().detach()
                    output = net(data)
                    right = rightness(output, target)
                    val_rights.append(right)

                # 获取在训练集上的正确的数量和样本数量
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                # 获取在验证集上的正确的数量和样本数量
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

                # 打印验证集和训练集情况
                print(val_r)
                print('训练周期: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t训练正确率: {:.2f}%\t校验正确率: {:.2f}%'.format(
                    epoch,
                    batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data,
                    100. * train_r[0].numpy() / train_r[1],
                    100. * val_r[0].numpy() / val_r[1]))

                # 记录训练集和验证集的错误率
                record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))

                # 记录训练过程中权重
                weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(),
                                net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])

    plt.plot(record)
    plt.xlabel('Steps')
    plt.ylabel('Error rate')
    plt.show()

    # 关闭dropout标记
    net.eval()
    # 记录准确率
    vals = []

    # 在测试集上进行测试
    for data, target in test_loader:
        data, target = data.clone(), target.clone().detach()
        output = net(data)
        val = rightness(output, target)
        vals.append(val)

    # 计算精确度
    rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
    right_rate = 100. * rights[0].numpy() / rights[1]
    print('测试集的精度：', right_rate)