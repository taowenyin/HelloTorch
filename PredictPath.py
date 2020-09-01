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
import matplotlib.pyplot as plt

from sklearn import linear_model
from docopt import docopt


if __name__ == '__main__':
    arguments = docopt(__doc__)

    # 数据集路径
    datasets_path = './data/Path/situation_0901.xlsx'

    # 计算数据集Sheet标签
    file = pd.ExcelFile(datasets_path)
    sheets_name = file.sheet_names

    # 构建完整数据集
    df = pd.read_excel(datasets_path, sheet_name=sheets_name)
    dataset = []
    lon_lat_data_y = []
    for v in df:
        data_item = df[v]
        data_item = data_item[data_item.keys().to_numpy()].to_numpy()
        # 经纬度时序数据
        lon_lat_data_y_item = data_item[:, 9:11]
        dataset.append(data_item)
        lon_lat_data_y.append(np.array(lon_lat_data_y_item, dtype=np.float))
    dataset = np.array(dataset, dtype=np.object)
    lon_lat_data_y = np.array(lon_lat_data_y, dtype=np.object)
    lon_lat_data_x = []
    # 构建时序标签
    for i in range(lon_lat_data_y.shape[0]):
        data_len = lon_lat_data_y[i].shape[0]
        data_y = np.arange(0, data_len)
        lon_lat_data_x.append(data_y)
    lon_lat_data_x = np.array(lon_lat_data_x, dtype=np.object)

    if arguments["--pre"]:
        # 表格纵轴
        pred_y_index = []
        pred_data = []
        pred_step = 20

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
            start = lon_lat_data_x_item.shape[0]
            lon_lat_data_pred = []
            # 计算后50步的预测值
            for i in range(pred_step):
                lon_lat_data_x_item_pred = np.array(lon_lat_data_x_item.shape[0] + i).reshape(-1, 1)
                lon_lat_data_y_item_pred = line_reg.predict(lon_lat_data_x_item_pred)
                lon_lat_data_item_pred_data = lon_lat_data_y_item_pred[0]
                lon_lat_data_pred.append(lon_lat_data_item_pred_data)
            lon_lat_data_item_coef_data = line_reg.coef_.flatten()
            lon_lat_data_item_intercept_data = line_reg.intercept_
            lon_lat_data_pred = np.array(lon_lat_data_pred).flatten()

            # 构建保存数据
            pred_data.append(np.hstack((np.hstack((lon_lat_data_pred, lon_lat_data_item_coef_data)),
                                        lon_lat_data_item_intercept_data)))
            pred_y_index.append(name)

        # 创建数据的列标题
        columns_name = []
        for i in range(pred_step):
            columns_name.append('longitude-{0}'.format(i + 1))
            columns_name.append('latitude-{0}'.format(i + 1))
        columns_name.append('coef_1')
        columns_name.append('coef_2')
        columns_name.append('intercept_1')
        columns_name.append('intercept_2s')

        # 图表索引
        plot_index = 1
        for i in range(len(pred_data)):
            # 预测数据
            pred_data_item = pred_data[i]
            pred_data_item = np.delete(pred_data_item,
                                       [len(pred_data_item) - 4, len(pred_data_item) - 3,
                                        len(pred_data_item) - 2, len(pred_data_item) - 1])
            longitude_pred_data_item = pred_data_item[::2]
            latitude_pred_data_item = pred_data_item[1::2]

            # 数据
            data_item = lon_lat_data_y[i]
            data_item = data_item.flatten()
            longitude_data_item = data_item[::2]
            latitude_data_item = data_item[1::2]
            longitude_data_item = longitude_data_item[-pred_step:]
            latitude_data_item = latitude_data_item[-pred_step:]

            plt.subplot(2, 2, plot_index)
            plt.title('{0} Predict'.format(pred_y_index[i]))
            line = plt.plot(longitude_pred_data_item, latitude_pred_data_item, label='Predict', marker='o')
            plt.plot(longitude_data_item, latitude_data_item, label='History', linestyle='--',
                     color=line[0].get_color(), marker='^')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plot_index = plot_index + 1
            if ((i + 1) % 4) == 0 or (i + 1) == len(pred_data):
                plot_index = 1
                plt.show()

        # 创建数据
        df = pd.DataFrame(pred_data, index=pred_y_index, columns=columns_name)
        # 写入Excel文件
        df.to_excel('./out/pred_data.xlsx', sheet_name='pred_data')

        print('Predict Data Finish...')

    if arguments["--cor"]:
        # 相似性矩阵
        sim_arr = []
        # 表格纵轴
        sim_index = []
        count = 1
        for i in range(len(lon_lat_data_y)):
            lon_lat_data_item = lon_lat_data_y[i]
            sim_arr_item = []
            # 获取对象唯一编号
            name = dataset[i][0][0]
            sim_index.append(name)
            for k in range(len(lon_lat_data_y)):
                if i == k:
                    sim_arr_item.append(-9999)
                    continue

                # 获得要计算相关性的对象
                lon_lat_data_com = lon_lat_data_y[k]

                # 计算余弦相似性
                sum_sim = 0
                # 数据长短不一时取短的
                data_len = min(len(lon_lat_data_item), len(lon_lat_data_com))
                last_dis = None
                for j in range(data_len):
                    A = lon_lat_data_item[j]
                    B = lon_lat_data_com[j]
                    # 计算欧式距离
                    dis = np.linalg.norm(A - B)
                    if last_dis is None:
                        last_dis = dis
                    else:
                        # 计算距离的变化
                        diff = dis - last_dis
                        sum_sim = sum_sim + diff
                sim = sum_sim / data_len
                # 保存相似性数据
                sim_arr_item.append(sim)
                print('Correlation Compute {0}/{1}'.format(count, len(lon_lat_data_y) ** 2 - len(lon_lat_data_y)))
                count = count + 1
            # 保存相似性数据
            sim_arr.append(sim_arr_item)

        sim_arr = np.array(sim_arr)
        sim_arr = sim_arr.flatten()
        delete_index = []
        for i in range(len(sheets_name)):
            delete_index.append(i * (len(sheets_name) + 1))
        sim_arr_tmp = np.delete(sim_arr, delete_index)
        # 归一化数据
        sim_min, sim_max = np.min(sim_arr_tmp), np.max(sim_arr_tmp)
        # 限制数据小于
        sim_arr_tmp = (1 - (sim_arr_tmp - sim_min) / (sim_max - sim_min)) * 0.99

        # 插入自相关的值
        insert_index = []
        for i in range(len(sheets_name)):
            insert_index.append(i * len(sheets_name))
        sim_arr = np.insert(sim_arr_tmp, insert_index, 1)
        # 重新变为二维数组
        sim_arr = sim_arr.reshape(-1, len(sheets_name))

        # 显示相关性热力图
        fig, ax = plt.subplots()
        im = ax.imshow(sim_arr)
        plt.colorbar(im)
        # 绘制坐标轴
        ax.set_xticks(np.arange(len(sim_index)))
        ax.set_yticks(np.arange(len(sim_index)))
        ax.set_xticklabels(sim_index)
        ax.set_yticklabels(sim_index)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # 绘制热力图中的数值
        for i in range(len(sheets_name)):
            for j in range(len(sheets_name)):
                text = ax.text(j, i, '{:.3f}'.format(sim_arr[i, j]), ha='center', va='center', color="w")
        fig.tight_layout()
        plt.title("Object Correlation Heatmap")

        # 创建数据
        df = pd.DataFrame(sim_arr, index=sim_index, columns=sim_index)
        # 写入Excel文件
        df.to_excel('./out/sim_data.xlsx', sheet_name='sim_data')

        print('Calculate Similarity Finish...')
        # 显示所有图表
        plt.show()
