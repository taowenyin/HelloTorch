import torch
import numpy as np

from torch import nn


# 稀疏自编码的损失函数
class KLDivergence(nn.Module):
    '''
    稀疏自编码器损失函数
    kl_param: KL参数
    kl_coeff: KL系数
    '''
    def __init__(self, kl_param, kl_coeff):
        super(KLDivergence, self).__init__()
        self.kl_param = kl_param
        self.kl_coeff = kl_coeff

    def forward(self, data):
        # 计算得到KL参数
        kl_hat = torch.mean(torch.abs(data), dim=0).detach().numpy()

        with np.errstate(divide='ignore'):
            # 检查inf，并设置为0
            kl_term_1 = self.kl_param / kl_hat
            kl_term_1[np.isinf(kl_term_1)] = 0
            kl_term_2 = (1 - self.kl_param) / (1 - kl_hat)
            kl_term_2[np.isinf(kl_term_2)] = 0

            # 计算得到KL散度
            kl = self.kl_param * torch.log(torch.from_numpy(kl_term_1)) + \
                 (1 - self.kl_param) * torch.log(torch.from_numpy(kl_term_2))
            kl = kl.detach().numpy()
            kl[np.isinf(kl)] = 0

            # 计算KL散度正则项的值
            kl_loss = self.kl_coeff * torch.sum(torch.from_numpy(kl))

            return kl_loss

