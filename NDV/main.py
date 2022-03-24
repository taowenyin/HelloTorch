import matplotlib.pyplot as plt
import numpy as np

from scipy import stats


if __name__ == '__main__':
    # 创建[-5, 5]之间的100个数
    x_1 = np.linspace(-5, 5, 100)
    # 设置均值loc为1，方差scale为1的正态分布结果
    y_1 = 2 / 3 * stats.norm.pdf(x_1, loc=1, scale=1)

    # 创建[-5, 5]之间的100个数
    x_2 = np.linspace(-5, 5, 100)
    # 设置均值loc为0，方差scale为0.25的正态分布结果
    y_2 = 1 / 3 * stats.norm.pdf(x_2, loc=0, scale=0.25)

    # 创建[-5, 5]之间的100个数
    x_3 = np.linspace(-5, 5, 100)
    y_3 = 0.5 * np.add(y_1, y_2)

    plt.grid()
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.title('Probability Density Function')
    plt.xticks(ticks=np.arange(-5, 5))
    plt.plot(x_1, y_1, label='G1')
    plt.plot(x_2, y_2, label='G2')
    plt.plot(x_3, y_3, label='G3')
    plt.legend()

    plt.show()
