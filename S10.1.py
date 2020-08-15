import numpy as np
import torch

from model.RNNModel import RNNModel
from model.LSTMModel import LSTMModel
from torch import nn, optim


if __name__ == '__main__':
    train_set = []
    valid_set = []

    samples = 2000

    sz = 10
    probability = 1.0 * np.array([10, 6, 4, 3, 1, 1, 1, 1, 1, 1])
    probability = probability[:sz]
    probability = probability / sum(probability)

    for m in range(samples):
        n = np.random.choice(range(1, sz + 1), p=probability)
        inputs = [0] * n + [1] * n
        inputs.insert(0, 3)
        inputs.append(2)
        train_set.append(inputs)

    for m in range(samples // 10):
        n = np.random.choice(range(1, sz + 1), p=probability)
        inputs = [0] * n + [1] * n
        inputs.insert(0, 3)
        inputs.append(2)
        valid_set.append(inputs)

    for m in range(2):
        n = sz + m
        inputs = [0] * n + [1] * n
        inputs.insert(0, 3)
        inputs.append(2)
        valid_set.append(inputs)

    np.random.shuffle(valid_set)

    lstm = LSTMModel(input_size=4, hidden_size=2, output_size=3, num_layers=1)
    # 交叉熵
    criterion = nn.NLLLoss()
    # 优化算法
    optimizer = optim.Adam(lstm.parameters(), lr=0.001)

    num_epoch = 50
    results = []
    for epoch in range(num_epoch):
        train_loss = 0
        np.random.shuffle(train_set)
        for i, seq in enumerate(train_set):
            loss = 0
            hidden = lstm.init_hidden()
            for t in range(len(seq) - 1):
                x = torch.LongTensor([seq[t]]).unsqueeze(0)
                y = torch.LongTensor([seq[t + 1]])
                output, hidden = lstm(x, hidden)
                loss += criterion(output, y)
            loss = 1.0 * loss / len(seq)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss

            if i > 0 and i % 500 == 0:
                print('第{}轮，第{}个，训练Loss: {:0.2f}'.format(epoch, i, train_loss.data.numpy() / i))

    valid_loss = 0
    errors = 0
    show_out = ''
    for i, seq in enumerate(valid_set):
        loss = 0
        outstring = ''
        targets = ''
        diff = 0
        hidden = lstm.init_hidden()
        for t in range(len(seq) - 1):
            x = torch.LongTensor([seq[t]]).unsqueeze(0)
            y = torch.LongTensor([seq[t + 1]])
            output, hidden = lstm(x, hidden)
            mm = torch.max(output, 1)[1][0]
            outstring += str(mm.data.numpy())
            targets += str(y.data.numpy()[0])
            loss += criterion(output, y)
            diff += 1 - mm.eq(y).data.numpy()[0]

        loss = 1.0 * loss / len(seq)
        valid_loss += loss
        errors += diff
        if np.random.rand() < 0.1:
            # 以0.1概率记录一个输出字符串
            show_out = outstring + '\n' + targets

    # 打印结果
    print(output[0][2].data.numpy())
    print('第{}轮, 训练Loss:{:.2f}, 校验Loss:{:.2f}, 错误率:{:.2f}'.format(epoch,
                                                                  train_loss.data.numpy() / len(train_set),
                                                                  valid_loss.data.numpy() / len(valid_set),
                                                                  1.0 * errors / len(valid_set)
                                                                  ))

    print(show_out)
    results.append([train_loss.data.numpy() / len(train_set),
                    valid_loss.data.numpy() / len(train_set),
                    1.0 * errors / len(valid_set)
                    ])