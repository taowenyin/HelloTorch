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
    # 读取变量x
    x = torch.FloatTensor(np.arange(len(counts), dtype=float))
    # 读取单车数量Y
    y = torch.FloatTensor(np.array(counts, dtype=float))

    # 设置隐藏层神经元数量
    sz = 10
    # 初始化输入层到隐藏层的权重矩阵，尺寸是(1, 10)
    i_h_weight = torch.randn(1, sz).requires_grad_(True)
    # 初始化隐藏层节点的扁置向量
    biases = torch.randn(sz).requires_grad_(True)
    # 初始化从隐藏层到输出曾的权重矩阵，尺寸是(10, 1)
    h_o_weight = torch.randn(sz, 1).requires_grad_(True)

    # 设置学习率
    learning_rate = 0.0001
    # 记录每次迭代的损失值
    losses = []

    for i in range(1000000):
        # 从输入层到隐藏层的计算，得到隐藏层(50, 10)的值
        hidden = x.expand(sz, len(x)).t() * i_h_weight.expand(len(x), sz) + biases.expand(len(x), sz)
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
        i_h_weight.grad.zero_()
        biases.grad.zero_()
        h_o_weight.grad.zero_()

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
