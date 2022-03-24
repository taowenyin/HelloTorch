import os
import numpy as np
import matplotlib.pyplot as plt


#计算两个点之间的欧几里得距离
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


 # 得到一个点到数据集里面所有点的距离
def get_euclidean_distance(point, dataSet):
    distance_set = []
    for sample in dataSet:
        if euclidean_distance(point, sample): # 排除这个点本身，即排除与自己距离为0的点
            distance_set.append(euclidean_distance(point, sample))
    return distance_set


# k近邻算法
class KNN():
    k = 0

    def train(self, x_train, y_train):  # train函数得到最好的k值，即以周围多少个点进行投票效果最好
        Kmax = len(x_train)  # k最大为点的个数
        best_k = 0  # 初始化最好的k
        best_accurrcy = 0  # 初始化最好的k对应的准确度
        for k in range(1, 2):  # 依次计算每一个k
            labelSet = self.predict(x_train, k)  # 计算当前k下各点的label
            count = np.sum(np.logical_xor(labelSet, y_train) == 1)  # 预测结果与真实标记不一致就说明预测失败
            precision = 1 - count / y_train.shape[0]  # 计算在训练集上的准确度
            print("k = %2d accurrcy: %.2f" % (k, precision))
            if precision > best_accurrcy:  # 记录最好的k
                best_accurrcy = precision
                best_k = k
        return best_k, best_accurrcy

    def predict(self, predictSet, k):
        labelSet = []
        for point in predictSet:
            distance_set = get_euclidean_distance(point, x_train)  # 得到当前点与训练集所有点的距离
            # 得到距离从小到大排序的索引
            sorted_index = sorted(range(len(distance_set)), key=lambda k: distance_set[k], reverse=False)

            # 计算前k个最近的点的label个数，进行投票
            label_count = [list(y_train[sorted_index[:k]]).count(0),
                           list(y_train[sorted_index[:k]]).count(1),
                           list(y_train[sorted_index[:k]]).count(2)]
            labelSet.append(label_count.index(max(label_count)))

        return labelSet


# 画出决策边界
def plot_desicion_boundary(X, y, knn):
    x_min = np.array(X)[:, 0].min() - 1 # 计算图的边界
    x_max = np.array(X)[:, 0].max() + 1
    y_min = np.array(X)[:, 1].min() - 1
    y_max = np.array(X)[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = knn.predict(np.vstack([xx.ravel(), yy.ravel()]).T.tolist(), knn.k)
    Z = np.array(Z).reshape(xx.shape)
    f, axarr = plt.subplots(1, 1)
    axarr.contourf(xx, yy, Z, alpha = 0.4)
    axarr.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=y, s=10, edgecolor='k')
    axarr.set_title("KNN (k={})".format(1))
    plt.show()


if __name__ == '__main__':
    print('load data...')
    path = os.path.abspath(os.path.dirname(__file__))  # 获取py文件当前路径
    x_train = np.loadtxt(path + "/w1w2w3.csv", delimiter=",", usecols=(0, 1))
    y_train = np.loadtxt(path + "/w1w2w3.csv", delimiter=",", usecols=(2))
    print('finish data load...')  # 读取数据

    knn = KNN()  # 实例化KNN模型
    best_k, best_accurrcy = knn.train(x_train, y_train)  # 得到最好的k
    print("best k =", best_k, " best accurrcy:", best_accurrcy)
    knn.k = best_k  # 记录最好的k
    plot_desicion_boundary(x_train, y_train, knn)  # 画出决策边界

