"""

Predict Path

Usage:
  PredictPath.py [--pre] [--cor]
  PredictPath.py (-h | --help)

Options:
  -h --help     Show this screen
  --pre         Predict Path
  --cor         Generate Correlation

"""


import pandas as pd
import numpy as np

from sklearn import linear_model
from docopt import docopt


if __name__ == '__main__':
    arguments = docopt(__doc__)

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
    lon_lat_data_y = np.array(dataset[:, :, 9:11], dtype=np.float)
    lon_lat_data_x = []
    # 构建时序标签
    for i in range(lon_lat_data_y.shape[0]):
        data_len = lon_lat_data_y.shape[1]
        data_y = np.arange(0, data_len)
        lon_lat_data_x.append(data_y)
    lon_lat_data_x = np.array(lon_lat_data_x)

    if arguments["--pre"]:
        # 保存预测值
        lon_lat_data_pred = {}
        # 表格纵轴
        pred_y_index = []
        pred_data = []

        for i in range(lon_lat_data_y.shape[0]):
            lon_lat_data_y_item = lon_lat_data_y[i]
            lon_lat_data_x_item = lon_lat_data_x[i].reshape(-1, 1)
            # 获取对象唯一编号
            name = dataset[i][0][0]

            # 构建线性回归模型
            line_reg = linear_model.LinearRegression()
            # 拟合现有数据
            line_reg.fit(lon_lat_data_x_item, lon_lat_data_y_item)

            # 数据预测
            lon_lat_data_x_item_pred = np.array(lon_lat_data_x_item.shape[0]).reshape(-1, 1)
            lon_lat_data_y_item_pred = line_reg.predict(lon_lat_data_x_item_pred)

            lon_lat_data_item_pred_data = lon_lat_data_y_item_pred[0]
            lon_lat_data_item_coef_data = line_reg.coef_.flatten()
            lon_lat_data_item_intercept_data = line_reg.intercept_

            # 保存预测相关数据
            pred_item = {'pred': lon_lat_data_item_pred_data,
                         'coef': lon_lat_data_item_coef_data,
                         'intercept': lon_lat_data_item_intercept_data}
            lon_lat_data_pred[name] = pred_item

            # 构建保存数据
            pred_data.append(np.hstack((np.hstack((lon_lat_data_item_pred_data, lon_lat_data_item_coef_data)),
                                        lon_lat_data_item_intercept_data)))
            pred_y_index.append(name)

        # 打印完整数据
        print(lon_lat_data_pred)
        # 创建数据
        df = pd.DataFrame(pred_data, index=pred_y_index,
                          columns=['longitude', 'latitude', 'coef_1', 'coef_2', 'intercept_1', 'intercept_1'])
        # 写入Excel文件
        df.to_excel('./out/pred_data.xlsx', sheet_name='pred_data')

    if arguments["--cor"]:
        for i in range(len(lon_lat_data_y)):
            lon_lat_data_item = lon_lat_data_y[i]
            for k in range(len(lon_lat_data_y)):
                if i == k:
                    continue

                # 获得要计算相关性的对象
                lon_lat_data_com = lon_lat_data_y[k]

                # 计算数据均值
                lon_lat_data_item_mu = np.mean(lon_lat_data_item, axis=1)
                lon_lat_data_com_mu = np.mean(lon_lat_data_com, axis=1)

                # 计算数据纬度
                data_dim = lon_lat_data_com.shape[1]

                # 计算数据的标准差
                std_item = np.std(lon_lat_data_item, axis=1, ddof=data_dim - 1)
                std_com = np.std(lon_lat_data_com, axis=1, ddof=data_dim - 1)

                cov = np.dot(lon_lat_data_item, lon_lat_data_com.T) - data_dim * np.dot(
                    lon_lat_data_item_mu[:, np.newaxis], lon_lat_data_com_mu[np.newaxis, :])

                cor = cov / np.dot(std_item[:, np.newaxis], std_com[np.newaxis, :])

                print('xxx')


        print('xxx')