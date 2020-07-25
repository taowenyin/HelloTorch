import os
import torch
import numpy as np
import utils.functions as functions
import utils.abide.prepare_utils as PrepareUtils

from torch import nn, optim
from model.AutoEncoderModel import AutoEncoderModel


def run_autoencoder_1(fold, input_layer, code_layers, output_layer,
                      train_loader, validation_loader, test_loader, model_path):
    """
    第一个自编码器
    :param fold: 数据折
    :param input_layer: 输入层的特征数
    :param code_layers: 编码层的特征数
    :param output_layer: 输出层的特征数
    :param train_loader: 训练数据
    :param validation_loader: 验证数据
    :param test_loader: 测试数据
    :param model_path: 保存模型的地址
    :return: 训练误差、验证误差、测试误差
    """

    # 模型初始化
    PrepareUtils.reset()

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

    # 一个超大的误差
    prev_error = 9999999999
    # 当前验证集误差
    curr_validation_error = 0

    if os.path.isfile(model_path):
        return None, None, None

    # 构建自编码器
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
            print('AE1 Fold {0} Epoch {1} Batch {2} Train Loss: {3}'.format(fold, epoch, batch_idx, loss))
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
                    print('AE1 Fold {0} Batch {1} Validation Loss: {2}'.format(fold, batch_idx, loss))
                    validation_error.append(loss.cpu())
                    curr_validation_error = loss.cpu()

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
            print('AE1 Fold {0} Batch {1} Test Loss: {2}'.format(fold, batch_idx, loss))
            test_error.append(loss.cpu())

        # 保存模型
        if curr_validation_error < prev_error:
            print('Save Better AE1 Model...')
            torch.save(model.state_dict(), model_path)
            prev_error = curr_validation_error

    return train_error, validation_error, test_error


def run_autoencoder_2(fold, prev_model_path, prev_input_layer, prev_code_layers, code_layers, output_layer,
                      train_loader, validation_loader, test_loader, model_path):
    """
    第二个自编码器
    :param fold: 折数
    :param prev_model_path: AE1模型的地址
    :param prev_input_layer: AE1模型的输入
    :param prev_code_layers: AE1模型的编码层
    :param code_layers: AE2模型的编码层
    :param output_layer: AE2模型的输出层
    :param train_loader: 训练数据
    :param validation_loader: 验证数据
    :param test_loader: 测试数据
    :param model_path: 保存AE2模型的地址
    :return: 训练误差、验证误差、测试误差
    """

    # 模型初始化
    PrepareUtils.reset()

    if torch.cuda.is_available():
        gpu_status = True
    else:
        gpu_status = False

    # 第一个自编码器的去噪率
    denoising_rate = 0.9
    # 自编码器1的学习率
    learning_rate = 0.0001
    # 训练周期
    EPOCHS = 10

    # 保存训练、验证误差
    train_error = []
    validation_error = []
    test_error = []

    # 一个超大的误差
    prev_error = 9999999999
    # 当前验证集误差
    curr_validation_error = 0

    if os.path.isfile(model_path):
        return None, None, None

    # 构建上一个自编码器
    prev_model = AutoEncoderModel(prev_input_layer, np.array([prev_code_layers]), prev_input_layer, is_denoising=False)
    # 读取保存的模型
    prev_model.load_state_dict(torch.load(prev_model_path))
    if gpu_status:
        prev_model = prev_model.cuda()

    # 构建自编码器
    model = AutoEncoderModel(prev_code_layers, code_layers, output_layer, is_denoising=True, denoising_rate=denoising_rate)
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
            # 把数据化为第一个编码器的编码层
            train_X, _ = prev_model.forward(data.cuda())
            # 前向传播，返回编码器和解码器
            encoder, decoder = model(train_X)
            # 获取误差，并添加正则项
            loss = criterion(decoder, train_X)
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 一步随机梯度下降算法
            optimizer.step()
            # 打印损失值
            print('AE2 Fold {0} Epoch {1} Batch {2} Train Loss: {3}'.format(fold, epoch, batch_idx, loss))
            train_error.append(loss.cpu())

            if batch_idx % 5 == 0 and batch_idx != 0:
                # 关闭反向传播
                model.eval()
                # 开始验证所有数据
                for batch_idx, (data, target) in enumerate(validation_loader):
                    if gpu_status:
                        data = data.cuda()
                    # 把数据化为第一个编码器的编码层
                    validation_X, _ = prev_model(data)
                    # 前向传播，返回编码器和解码器
                    encoder, decoder = model(validation_X)
                    # 获取误差，并添加正则项
                    loss = criterion(decoder, validation_X)
                    # 打印损失值
                    print('AE2 Fold {0} Batch {1} Validation Loss: {2}'.format(fold, batch_idx, loss))
                    validation_error.append(loss.cpu())
                    curr_validation_error = loss.cpu()

        # 关闭反向传播
        model.eval()
        # 开始验证所有数据
        for batch_idx, (data, target) in enumerate(test_loader):
            if gpu_status:
                data = data.cuda()
            # 把数据化为第一个编码器的编码层
            test_X, _ = prev_model(data)
            # 前向传播，返回编码器和解码器
            encoder, decoder = model(test_X)
            # 获取误差，并添加正则项
            loss = criterion(decoder, test_X)
            # 打印损失值
            print('AE2 Fold {0} Batch {1} Test Loss: {2}'.format(fold, batch_idx, loss))
            test_error.append(loss.cpu())

        # 保存模型
        if curr_validation_error < prev_error:
            print('Save Better AE2 Model...')
            torch.save(model.state_dict(), model_path)
            prev_error = curr_validation_error

    return train_error, validation_error, test_error


def run_finetuning(fold, prev_1_model_path, prev_2_model_path,
                   prev_1_input_layer, prev_1_code_layers, prev_2_code_layers,
                   train_loader, validation_loader, test_loader, model_path):
    # 多层感知机的学习率
    learning_rate = 0.0005
    # 第一层的dropout
    dropout_1 = 0.6
    # 第二层的dropout
    dropout_2 = 0.8
    # 开始的动量值
    initial_momentum = 0.1
    # 每次迭代增加动量，最终的动量值
    final_momentum = 0.9
    saturate_momentum = 100

    EPOCHS = 100

    start_saving_at = 20
    batch_size = 10
    n_classes = 2

    # 载入AE 1模型
    ae_1_model_param = PrepareUtils.load_ae_encoder(prev_1_input_layer, prev_1_code_layers, prev_1_model_path)
    # 载入AE 2模型
    ae_2_model_param = PrepareUtils.load_ae_encoder(prev_1_code_layers, prev_2_code_layers, prev_2_model_path)

    # 开始训练
    for epoch in range(EPOCHS):
        # 训练所有数据
        for batch_idx, (data, target) in enumerate(train_loader):
            # 输出进行one-hot编码
            train_y = np.array([PrepareUtils.to_softmax(n_classes, y) for y in target])

            if batch_idx % 5 == 0 and batch_idx != 0:
                # 开始验证所有数据
                for batch_idx, (data, target) in enumerate(validation_loader):
                    # 输出进行one-hot编码
                    validation_y = np.array([PrepareUtils.to_softmax(n_classes, y) for y in target])

        # 开始验证所有数据
        for batch_idx, (data, target) in enumerate(test_loader):
            # 输出进行one-hot编码
            test_y = np.array([PrepareUtils.to_softmax(n_classes, y) for y in target])

    return None