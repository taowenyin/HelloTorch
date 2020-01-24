import torch
import torch.nn as nn
import torch.optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
import re

from collections import Counter


# 替换所有标点符号
def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
    return sentence


# 构建词袋模型
def prepare_data(good_file, bad_file, is_filter=True):
    # 存储所有单词
    all_words = []
    # 存储正向评论
    pos_sentence = []
    # 存储负向评论
    neg_sentence = []

    with open(good_file, 'r') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                # 过滤标点符号
                line = filter_punc(line)
            # 分词
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                pos_sentence.append(words)
    print('{0} 包含 {1} 行, {2} 个词.'.format(good_file, idx + 1, len(all_words)))

    count = len(all_words)
    with open(bad_file, 'r') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                # 过滤标点符号
                line = filter_punc(line)
            # 分词
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                neg_sentence.append(words)
    print('{0} 包含 {1} 行, {2} 个词.'.format(bad_file, idx + 1, len(all_words) - count))

    # 建立词典，diction的每一项为{w:[id, 单词出现次数]}
    diction = {}
    cnt = Counter(all_words)
    for word, freq in cnt.items():
        diction[word] = [len(diction), freq]
    print('字典大小:{}'.format(len(diction)))
    return pos_sentence, neg_sentence, diction


# 获取单词的索引
def word2index(word, diction):
    if word in diction:
        value = diction[word][0]
    else:
        value = -1
    return value


# 根据索引获取单词
def index2word(index, diction):
    for w, v in diction.item():
        if v[0] == index:
            return w
    return None


# 把文本向量化
def sentence2vec(sentence, dictionary):
    vector = np.zeros(len(dictionary))
    for l in sentence:
        vector[l] += 1
    return 1.0 * vector / len(sentence)


# 分类准确度
def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


if __name__ == '__main__':
    good_file = 'data/good.txt'
    bad_file = 'data/bad.txt'

    pos_sentence, neg_sentence, diction = prepare_data(good_file, bad_file)

    dataset = []  # 数据集
    labels = []  # 标签集
    sentences = []  # 原始句子
    # 处理正向评论
    for sentence in pos_sentence:
        new_sentence = []
        for l in sentence:
            if l in diction:
                new_sentence.append(word2index(l, diction))
        dataset.append(sentence2vec(new_sentence, diction))
        labels.append(0)
        sentences.append(sentence)

    # 处理负向评论
    for sentence in neg_sentence:
        new_sentence = []
        for l in sentence:
            if l in diction:
                new_sentence.append(word2index(l, diction))
        dataset.append(sentence2vec(new_sentence, diction))
        labels.append(1)
        sentences.append(sentence)

    # 打乱后重新生成数据集
    indices = np.random.permutation(len(dataset))
    dataset = [dataset[i] for i in indices]
    labels = [labels[i] for i in indices]
    sentences = [sentences[i] for i in indices]

    # 生成数据集
    test_size = len(dataset) // 10
    # 训练数据集
    train_data = dataset[2 * test_size:]
    train_label = labels[2 * test_size:]
    # 验证数据集
    valid_data = dataset[:test_size]
    valid_label = labels[:test_size]
    # 测试数据集
    test_data = dataset[test_size : 2 * test_size]
    test_label = labels[test_size : 2 * test_size]

    # 构建模型
    model = nn.Sequential(nn.Linear(len(diction), 10), nn.ReLU(), nn.Linear(10, 2), nn.LogSoftmax(dim=1))
    # 定义损失函数为交叉熵
    cost = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    records = []
    losses = []

    for epoch in range(10):
        for i, data in enumerate(zip(train_data, train_label)):
            x = torch.tensor(data[0], requires_grad=True, dtype=torch.float).view(1, -1)
            y = torch.tensor(np.array([data[1]]), dtype=torch.long)

            optimizer.zero_grad()
            predict = model(x)
            loss = cost(predict, y)
            losses.append(loss.data.numpy())
            loss.backward()
            optimizer.step()

            if i % 3000 == 0:
                val_losses = []
                rights = []
                for j, val in enumerate(zip(valid_data, valid_label)):
                    x = torch.tensor(val[0], requires_grad=True, dtype=torch.float).view(1, -1)
                    y = torch.tensor(np.array([val[1]]), dtype=torch.long)
                    predict = model(x)
                    right = rightness(predict, y)
                    rights.append(right)
                    loss = cost(predict, y)
                    val_losses.append(loss.data.numpy())

                right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
                print('第{}轮，训练损失：{:.2f}, 校验损失：{:.2f}, 校验准确率: {:.2f}'.format(epoch, np.mean(losses),
                                                                            np.mean(val_losses), right_ratio))
                records.append([np.mean(losses), np.mean(val_losses), right_ratio])

    # 绘制误差曲线
    a = [i[0] for i in records]
    b = [i[1] for i in records]
    c = [i[2] for i in records]
    plt.plot(a, label='Train Loss')
    plt.plot(b, label='Valid Loss')
    plt.plot(c, label='Valid Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Loss & Accuracy')
    plt.legend()
    plt.show()