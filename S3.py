import torch
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
    train_data = dataset[2 * test_size:]
    train_label = labels[2 * test_size:]

    valid_data = dataset[:test_size]
    valid_label = labels[:test_size]

    test_data = dataset[test_size : 2 * test_size]
    test_label = labels[test_size : 2 * test_size]