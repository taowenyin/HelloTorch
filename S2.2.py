import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    data_path = 'data/hour.csv'
    # 读取数据文件
    rides = pd.read_csv(data_path)
    # 要进行one-hot编码的字段
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        # 取出所有类型变量，并进行one-hot编码
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        # 将新的编码变量与所有变量进行合并
        rides = pd.concat([rides, dummies], axis=1)

    # 要删除的字段名
    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    # 从数据表中删除
    data = rides.drop(fields_to_drop, axis=1)

    # 要进行数值标准化的字段，采用z-score方法使得数据符合正态分布
    quant_features = ['cnt', 'temp', 'hum', 'windspeed']
    # 保存每个变量的均值和方差
    scaled_features = {}
    for each in quant_features:
        # 计算均值和方差
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        # 把数据进行归一化
        data.loc[:, each] = (data[each] - mean)/std

    # 选出训练集
    train_data = data[:-21 * 24]
    # 选出测试集
    test_data = data[-21 * 24:]
    # 目标列的字段
    target_fields = ['cnt', 'casual', 'registered']

    # 获得训练集和测试集的特征列和目标列
    train_features, train_targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

    # 获取输入数据
    X = train_features.values
    Y = train_targets['cnt'].values
    Y = np.reshape(Y, [len(Y), 1])
    # 保存损失值
    losses = []

    # 定义一个神经网络
    input_size = train_features.shape[1]
    hidden_size = 10
    output_size = 1
    batch_size = 128
    neu = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_size, output_size)).to(device)
    cost = torch.nn.MSELoss()
    optimizer = optim.SGD(neu.parameters(), lr=0.01)

    for i in range(1000):
        batch_loss = []
        for start in range(0, len(X), batch_size):
            # 计算分批索引
            end = start + batch_size if start + batch_size < len(X) else len(X)
            xx = torch.tensor(X[start:end], dtype=torch.float, requires_grad=True).to(device)
            yy = torch.tensor(Y[start:end], dtype=torch.float, requires_grad=True).to(device)
            # 计算预测值
            predict = neu(xx)
            # 计算损失值
            loss = cost(predict, yy)
            # 清空梯度
            optimizer.zero_grad()
            # 反响传播
            loss.backward()
            optimizer.step()
            loss = loss.cpu()
            batch_loss.append(loss.data.numpy())

        if i % 100 == 0:
            losses.append(np.mean(batch_loss))
            print(i, np.mean(batch_loss))

    targets = test_targets['cnt']
    targets = targets.values.reshape([len(targets), 1])

    x = torch.tensor(test_features.values, dtype=torch.float, requires_grad=True).to(device)
    y = torch.tensor(targets, dtype=torch.float, requires_grad=True).to(device)

    # 执行训练好的网络进行预测
    predict = neu(x).cpu()
    predict = predict.data.numpy()

    mean, std = scaled_features['cnt']
    plt.plot(np.arange(len(y)), predict * std + mean, '--', label='Prediction')
    plt.plot(np.arange(len(y)), targets * std + mean, '-', label='Data')
    plt.legend()
    plt.xlabel('Date-time')
    plt.ylabel('Counts')
    plt.show()

    # fig, ax = plt.subplots(figsize=(10, 7))
    # mean, std = scaled_features['cnt']
    # ax.plot(predict * std + mean, label='Prediction', linestyle='--')
    # ax.plot(targets * std + mean, label='Data', linestyle='-')
    # ax.legend()
    # ax.set_xlabel('Date-time')
    # ax.set_ylabel('Counts')
    # # 对横坐标轴进行标注
    # dates = pd.to_datetime(rides.loc[test_data.index]['dteday'])
    # dates = dates.apply(lambda d: d.strftime('%b %d'))
    # ax.set_xticks(np.arange(len(dates))[12::24])
    # _ = ax.set_xticklabels(dates[12::24], rotation=45)
