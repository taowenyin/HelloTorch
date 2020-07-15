import torch

from torch import nn


# 稀疏自编码的损失函数
class SparseAeLoss(nn.Module):
    '''
    稀疏自编码器损失函数
    sparse_param: 稀疏参数
    sparse_coeff: 稀疏系数
    '''
    def __init__(self, sparse_param, sparse_coeff):
        super(SparseAeLoss, self).__init__()
        self.sparse_param = sparse_param
        self.sparse_coeff = sparse_coeff

    def forward(self, prediction, target):
        # 均方差
        mse_loss = nn.MSELoss()
        # kl散度
        kl_loss = nn.KLDivLoss()
        # 获取均方差损失的误差
        mse_loss_output = mse_loss(prediction, target)

        # 获取KL损失的误差
        # todo 要验证、修改
        kl_loss_output = kl_loss(prediction, target)

        return torch.from_numpy(mse_loss_output + self.sparse_coeff * kl_loss_output)

