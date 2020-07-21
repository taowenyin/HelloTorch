import torch
import utils.functions as functions

from torch import nn, optim
from model.AutoEncoderModel import AutoEncoderModel


# 第一个自编码器
def run_autoencoder_1(fold, input_layer, code_layers, output_layer, train_loader, validation_loader, test_loader):
    if torch.cuda.is_available():
        gpu_status = True
    else:
        gpu_status = False

    # 第一个自编码器的去噪率
    denoising_rate = 0.7
    # 自编码器1的学习率
    learning_rate = 0.0001
    # 稀疏参数
    sparse_param = 0.2
    # 稀疏系数
    sparse_coeff = 0.5
    # 训练周期
    EPOCHS = 10

    # 保存训练、验证误差
    train_error = []
    validation_error = []
    test_error = []

    # 构建自编码器1和自编码器2
    model = AutoEncoderModel(input_layer, code_layers, output_layer, is_denoising=True, denoising_rate=denoising_rate)
    if gpu_status:
        model = model.cuda()
    # 使用随机梯度下降进行优化
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 使用均方差作为损失函数
    criterion = nn.MSELoss()

    # 开始训练
    for epoch in range(EPOCHS):
        # 打开反向传播
        model.train()
        # 训练所有数据
        for batch_idx, (data, target) in enumerate(train_loader):
            if gpu_status:
                data = data.cuda()
            # 前向传播，返回编码器和解码器
            encoder, decoder = model(data)
            # 获取误差，并添加正则项
            loss = criterion(decoder, data)
            # 计算KL散度
            penalty = functions.kl_divergence(encoder.cpu().detach().numpy(), sparse_param, sparse_coeff)
            loss = loss + penalty
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 一步随机梯度下降算法
            optimizer.step()
            # 打印损失值
            print('Fold {0} Epoch {1} Batch {2} Train Loss: {3}'.format(fold, epoch, batch_idx, loss))
            train_error.append(loss.cpu())

            if batch_idx % 5 == 0 and batch_idx != 0:
                # 关闭反向传播
                model.eval()
                # 开始验证所有数据
                for batch_idx, (data, target) in enumerate(validation_loader):
                    if gpu_status:
                        data = data.cuda()
                    # 前向传播，返回编码器和解码器
                    encoder, decoder = model(data)
                    # 获取误差，并添加正则项
                    loss = criterion(decoder, data)
                    # 计算KL散度
                    penalty = functions.kl_divergence(encoder.cpu().detach().numpy(), sparse_param, sparse_coeff)
                    loss = loss + penalty
                    # 打印损失值
                    print('Fold {0} Batch {1} Validation Loss: {2}'.format(fold, batch_idx, loss))
                    validation_error.append(loss.cpu())

        # 关闭反向传播
        model.eval()
        # 开始验证所有数据
        for batch_idx, (data, target) in enumerate(test_loader):
            if gpu_status:
                data = data.cuda()
            # 前向传播，返回编码器和解码器
            encoder, decoder = model(data)
            # 获取误差，并添加正则项
            loss = criterion(decoder, data)
            # 计算KL散度
            penalty = functions.kl_divergence(encoder.cpu().detach().numpy(), sparse_param, sparse_coeff)
            loss = loss + penalty
            # 打印损失值
            print('Fold {0} Batch {1} Test Loss: {2}'.format(fold, batch_idx, loss))
            test_error.append(loss.cpu())

    return train_error, validation_error, test_error