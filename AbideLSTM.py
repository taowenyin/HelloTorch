"""

LSTM training and fine-tuning.

Usage:
  nn.py [--whole] [--male] [--threshold] [--leave-site-out] [<derivative> ...]
  nn.py (-h | --help)

Options:
  -h --help           Show this screen
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  derivative          Derivatives to process

"""


import numpy as np
import torch
import utils.abide.prepare_utils as PrepareUtils
import time

from docopt import docopt
from model.LSTMModel import LSTMModel
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data.ABIDE.AbideData import AbideData


# 不定长数据集的预处理
def collate_fn(data):
    batch_data_x = []
    batch_data_y = []
    batch_data_length = []

    for v in data:
        # 添加数据长度
        batch_data_length.append(v[0].shape[0])
        # 添加数据
        batch_data_x.append(v[0])
        # 添加标签
        batch_data_y.append(v[1])

    # 增加数据padding
    batch_data_x_pad = nn.utils.rnn.pad_sequence(batch_data_x, batch_first=True, padding_value=0)
    # 增加数据pack
    batch_data_x_pack = nn.utils.rnn.pack_padded_sequence(batch_data_x_pad,
                                                          batch_data_length, batch_first=True, enforce_sorted=False)

    return batch_data_x_pack, torch.from_numpy(np.array(batch_data_y))


if __name__ == '__main__':
    # 开始计时
    start = time.process_time()

    arguments = docopt(__doc__)

    if torch.cuda.is_available():
        gpu_status = True
    else:
        gpu_status = False

    # 表型数据位置
    pheno_path = './data/ABIDE/phenotypes/Phenotypic_V1_0b_preprocessed1.csv'
    # 载入表型数据
    pheno = PrepareUtils.load_phenotypes(pheno_path)
    # 载入数据集
    hdf5 = PrepareUtils.hdf5_handler(bytes('./data/ABIDE/abide_lstm.hdf5', encoding='utf8'), 'a')

    # 脑图谱的选择
    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative
                   in arguments["<derivative>"]
                   if derivative in valid_derivatives]

    # 保存所有数据
    dataset_x = []
    dataset_y = []
    batch_size = 16
    # 训练周期
    EPOCHS = 50
    # 学习率
    learning_rate = 0.001
    # LSTM隐藏层数量
    lstm_hidden_num = 64
    # LSTM输出层数量
    lstm_output_num = 2
    # LSTM层数量
    lstm_layers_num = 2

    # 构建完整数据集
    hdf5_dataset = hdf5["patients"]
    for i in hdf5_dataset.keys():
        data_item_x = torch.from_numpy(np.array(hdf5["patients"][i]['cc200']))
        data_item_y = hdf5["patients"][i].attrs["y"]
        dataset_x.append(data_item_x)
        dataset_y.append(data_item_y)
    train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, test_size=0.3, shuffle=True)
    abideData_train = AbideData(train_x, train_y)
    abideData_test = AbideData(test_x, test_y)
    train_loader = DataLoader(dataset=abideData_train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=abideData_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 创建LSTM模型
    model = LSTMModel(train_x[0].shape[1], lstm_hidden_num, lstm_output_num, lstm_layers_num)
    if gpu_status:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 开启训练
    model.train()
    total_step = len(train_loader)
    for epoch in range(EPOCHS):
        for i, (data_x, data_y) in enumerate(train_loader):
            if gpu_status:
                data_x = data_x.float().cuda()
                data_y = data_y.cuda()
            output, (hidden_n, cell_n) = model(data_x)
            loss = criterion(output, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Train Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, EPOCHS, i + 1, total_step, loss))

    # 关闭反向传播
    model.eval()
    total_step = len(test_loader)
    correct = 0
    total = 0
    for i, (data_x, data_y) in enumerate(test_loader):
        if gpu_status:
            data_x = data_x.float().cuda()
            data_y = data_y.cuda()
        output, (hidden_n, cell_n) = model(data_x)
        # 获得预测值
        _, predicted = torch.max(output.data, 1)
        total += data_y.size(0)
        correct += (predicted == data_y).sum().item()

    print('Test Accuracy of the model on the test data: {:.2f} %'.format(100 * correct / total))

    end = time.process_time()
    print('Running time: {:.2f} Seconds'.format((end - start)))