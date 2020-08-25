import pandas as pd
import numpy as np


if __name__ == '__main__':
    # 计算数据集Sheet标签
    # index_range = np.arange(1, 115)
    index_range = np.arange(1, 3)
    sheets_name = []
    for v in index_range:
        sheets_name.append('Sheet{0}'.format(v))

    # 构建完整数据集
    df = pd.read_excel('./data/Path/situation_0821.xlsx', sheet_name=sheets_name)
    dataset = []
    for v in df:
        data_item = df[v]
        data_item = data_item[data_item.keys().to_numpy()].to_numpy()
        dataset.append(data_item)
    dataset = np.array(dataset)

    # 经纬度时序数据
    lon_lat_data_y = dataset[:, :, 9:11]
    lon_lat_data_x = []
    # 构建时序标签
    for i in range(lon_lat_data_y.shape[0]):
        data_len = lon_lat_data_y.shape[1]
        data_y = np.arange(0, data_len)
        lon_lat_data_x.append(data_y)
    lon_lat_data_x = np.array(lon_lat_data_x)

    print('xx')
    print('xx')