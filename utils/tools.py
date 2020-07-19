import torch


def random_uniform(shape, minval, maxval):
    '''产生某个范围内的均匀分布的随机数

    :param shape:形状
    :param minval:最大值
    :param maxval:最小值
    :return:返回shape形状的随机值
    '''
    return torch.rand(shape) * (maxval - minval) + minval