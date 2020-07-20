import torch
import numpy as np


def random_uniform(shape, minval, maxval):
    '''
    产生某个范围内的均匀分布的随机数
    :param shape:形状
    :param minval:最大值
    :param maxval:最小值
    :return:返回shape形状的随机值
    '''
    return torch.rand(shape) * (maxval - minval) + minval


def kl_divergence(data, kl_param, kl_coeff):
    '''
    计算KL散度
    :param data: 数据
    :param kl_param: KL参数
    :param kl_coeff: KL稀疏
    :return: KL散度值
    '''
    # 计算得到KL参数
    kl_hat = np.mean(np.abs(data), axis=0)

    # 忽略除数为0警告
    with np.errstate(divide='ignore'):
        # 检查inf，并设置为0
        kl_term_1 = kl_param / kl_hat
        kl_term_1[np.isinf(kl_term_1)] = 0
        kl_term_2 = (1 - kl_param) / (1 - kl_hat)
        kl_term_2[np.isinf(kl_term_2)] = 0

        # 计算得到KL散度
        kl = kl_param * np.log(kl_term_1) + (1 - kl_param) * np.log(kl_term_2)
        kl[np.isinf(kl)] = 0

        # 计算KL散度正则项的值
        kl_loss = kl_coeff * np.sum(kl)

        return kl_loss