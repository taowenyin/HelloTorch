import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_path = 'data/hour.csv'
    # 读取数据文件
    rides = pd.read_csv(data_path)
    # 读取前50个数据
    counts = rides['cnt'][:50]
    # 读取变量x，并且把x进行归一化
    x = torch.tensor(np.arange(len(counts), dtype=float) / len(counts), requires_grad=True).view(50, 1)
    # 读取单车数量Y
    y = torch.tensor(np.array(counts, dtype=float), requires_grad=True).view(50, 1)

    # 设置隐藏层神经元数量
    sz = 10
    # 初始化输入层到隐藏层的权重矩阵，尺寸是(1, 10)
    i_h_weight = torch.randn((1, sz), dtype=torch.double, requires_grad=True)
    # 初始化隐藏层节点的扁置向量
    biases = torch.randn(sz, dtype=torch.double, requires_grad=True)
    # 初始化从隐藏层到输出曾的权重矩阵，尺寸是(10, 1)
    h_o_weight = torch.randn((sz, 1), dtype=torch.double, requires_grad=True)

    # 设置学习率
    learning_rate = 0.001
    # 记录每次迭代的损失值
    losses = []
    # 记录最终预测值
    predictions = torch.tensor((len(counts), 1))

    for i in range(100000):
        # 从输入层到隐藏层的计算，得到隐藏层(50, 10)的值
        hidden = x * i_h_weight + biases
        # 使用sigmoid作为激活函数
        hidden = torch.sigmoid(hidden)
        # 从隐藏层输出到输出层，得到最终的预测值(50, 1)
        predictions = hidden.mm(h_o_weight)
        # 计算损失函数，通过数据比较与y比较，计算均方差
        loss = torch.mean((predictions - y) ** 2)
        # 保存损失值
        losses.append(loss.data.numpy())

        # 每隔10000个周期打印损失函数的值
        if i % 10000 == 0:
            print('loss:', loss)

        # 对损失函数进行反向传播计算梯度
        loss.backward()

        # 更新权重和偏置
        i_h_weight.data.add_(- learning_rate * i_h_weight.grad.data)
        biases.data.add_(- learning_rate * biases.grad.data)
        h_o_weight.data.add_(- learning_rate * h_o_weight.grad.data)

        # 清空梯度
        i_h_weight.grad.data.zero_()
        biases.grad.data.zero_()
        h_o_weight.grad.data.zero_()

    # 读取代预测的50个数据
    counts_predict = rides['cnt'][50:100]
    # 创建预测数据的x、y
    x = torch.tensor((np.arange(len(counts_predict), dtype = float) + len(counts) / len(counts_predict)), requires_grad=True).view(50, 1)
    y = torch.tensor(np.array(counts_predict, dtype=float), requires_grad=True).view(50, 1)
    # 从输入层到隐藏层的计算，得到隐藏层(50, 10)的值
    hidden = x * i_h_weight + biases
    hidden = torch.sigmoid(hidden)
    # 计算最终预测
    predictions = hidden.mm(h_o_weight)
    # 计算测试数据的损失值
    loss = torch.mean((predictions - y) ** 2)
    print(loss)

    plt.plot(x.data.numpy(), y.data.numpy(), 'o', label='Data')
    plt.plot(x.data.numpy(), predictions.data.numpy(), label='Prediction')
    plt.legend()
    plt.show()