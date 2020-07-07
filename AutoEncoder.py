import torch

from utils.data.MnistData import MnistData
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn, optim

from model.AutoEncoderModel import AutoEncoderModel, AutoEncoderType

if __name__ == '__main__':
    # 每批数据的大小
    batch_size = 100

    # 创建数据
    dataset = MnistData()
    train_labels, train_data = dataset.parse_train_data()
    test_labels, test_data = dataset.parse_test_target()
    validation_labels, validation_data = dataset.parse_validation_data()

    # 创建训练数据集
    train_dataset = TensorDataset(
        train_data.float().clone().detach().requires_grad_(True),
        train_labels.float().clone().detach().requires_grad_(True))
    # 创建训练数据集
    test_dataset = TensorDataset(
        test_data.float().clone().detach().requires_grad_(True),
        test_labels.float().clone().detach().requires_grad_(True))
    # 创建验证数据集
    validation_dataset = TensorDataset(
        validation_data.float().clone().detach().requires_grad_(True),
        validation_labels.float().clone().detach().requires_grad_(True))

    # 创建训练数据加载器，并且设置每批数据的大小，以及每次读取数据时随机打乱数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # 创建验证集加载器
    validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    # 测试集加载器
    test_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    model = AutoEncoderModel(28, 28, AutoEncoderType.Multi)
    # 均方差损失函数
    criterion = nn.MSELoss()
    # 采用Adam优化
    optimizier = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)