# 数据接口类
class IDataParse:
    def __init__(self, data_path=None):
        self._data_path = data_path

    '''
    获取训练数据
    返回：(labels, data)
    '''
    def parse_train_data(self):
        pass

    '''
    获取验证数据
    返回：(labels, data)
    '''
    def parse_validation_data(self):
        pass

    '''
    获取测试数据
    返回：(labels, data)
    '''
    def parse_test_target(self):
        pass