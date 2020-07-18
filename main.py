import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.arange(20, 30)
    # print(range(len(x)))

    plt.plot(range(len(x)), x)
    plt.show()