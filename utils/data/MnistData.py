from utils.data.IDataParse import IDataParse
from torchvision import datasets, transforms


class MnistData(IDataParse):
    '''
    :parameter ratio: 测试集的比例，剩下的就是验证集
    '''
    def __init__(self, data_path=None, ratio=0.8):
        super().__init__(data_path)

        # 将像素点转换到[-1, 1]之间，使得输入变成一个比较对称的分布，训练容易收敛
        data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        # 下载手写数据集的训练数据集
        self.__train_data = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
        # 获取手写数据集的测试数据集
        test_data = datasets.MNIST(root='./data', train=False, transform=data_tf)

        test_data_labels = test_data.test_labels
        test_data_data = test_data.test_data
        # 计算验证集和测试集的比例
        test_indices = int(len(test_data) * ratio)

        self.__test_data = test_data_data[:test_indices]
        self.__test_labels = test_data_labels[:test_indices]

        self.__validation_data = test_data_data[test_indices:]
        self.__validation_labels = test_data_labels[test_indices:]

    def parse_train_data(self):
        return self.__train_data.train_data, self.__train_data.train_labels

    def parse_validation_data(self):
        return self.__validation_data, self.__validation_labels

    def parse_test_target(self):
        return self.__test_data, self.__test_labels
