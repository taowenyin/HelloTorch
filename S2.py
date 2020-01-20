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
    x = torch.IntTensor(np.arange(len(counts)))
    # 读取单车数量Y
    y = torch.IntTensor(np.array(counts))

