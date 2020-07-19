import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = torch.linspace(0, 100).type(torch.IntTensor)
    rand = torch.randn(100) * 10
    y = x + rand

    # 创建训练集数据
    x_train = x[:-10]
    y_train = y[:-10]
    # 创建测试集数据
    x_test = x[-10:]
    y_test = y[-10:]

    # plt.scatter(x_train.data.numpy(), y_train.data.numpy())
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()

    # 模型：y = ax + b
    # 损失函数：(y - y^)^2 / N

    # 创建模型的两个变量
    a = torch.rand(1).requires_grad_(True)
    b = torch.rand(1).requires_grad_(True)
    # 初始化学习率
    learning_rate = 0.0001

    for i in range(1000):
        # 根据模型获得相关预测值
        predictions = a.expand_as(x_train) * x_train + b.expand_as(x_train)
        # 根据损失函数计算损失值
        loss = torch.mean((predictions - y_train) ** 2)
        # print('Loss:', regularization)
        # 反向传播计算梯度
        loss.backward()
        # 根据梯度更新变量
        a.data.add_(-learning_rate * a.grad.data)
        b.data.add_(-learning_rate * b.grad.data)
        # 清空梯度值
        a.grad.data.zero_()
        b.grad.data.zero_()

    # 模型结果
    fun_str = str(a.data.numpy()[0]) + 'x + ' + str(b.data.numpy()[0])
    # 训练数据的X轴
    x_data = x_train.data.numpy()
    # 测试数据的X轴
    x_pred = x_test.data.numpy()
    # 绘制训练数据实际值
    plt.plot(x_data, y_train.data.numpy(), 'o', label='Train')
    # 绘制测试数据实际值
    plt.plot(x_pred, y_test.data.numpy(), 's', label='Test')
    # 拼接训练与预测数据
    x_data = np.r_[x_data, x_test.data.numpy()]
    # 绘制训练数据预测值
    plt.plot(x_data, a.data.numpy() * x_data + b.data.numpy(), label=fun_str)
    # 绘制测试数据预测值
    plt.plot(x_pred, a.data.numpy() * x_pred + b.data.numpy(), 'o', label=fun_str)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

