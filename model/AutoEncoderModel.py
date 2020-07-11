import torch

from torch import nn
from enum import Enum

# https://blog.csdn.net/h__ang/article/details/90720579


# 定义自编码器类型
class AutoEncoderType(Enum):
    Single = 0  # 单层自编码器
    Multi = 1 # 多层自编码器
    Denoising = 2 # 去噪自编码器
    Sparse = 3 # 稀疏自编码器


class AutoEncoderModel(nn.Module):
    '''
    自编码器对象
    '''
    def __init__(self, image_width, image_height, autoencoder_type = AutoEncoderType.Single):
        super(AutoEncoderModel, self).__init__()

        # 判断输入的自编码器类型是否正确
        if not (type(autoencoder_type) is AutoEncoderType):
            print('Autoencoder type Error')
            return

        # 创建多层自编码器
        if autoencoder_type == AutoEncoderType.Multi:
            # 创建编码器
            self.encoder = nn.Sequential(
                nn.Linear(image_width * image_height, 128),
                nn.ReLU(True),
                nn.Linear(128, 64),
                nn.ReLU(True),
                nn.Linear(64, 12),
                nn.ReLU(True),
                nn.Linear(12, 3)
            )

            self.decoder = nn.Sequential(
                nn.Linear(3, 12),
                nn.ReLU(True),
                nn.Linear(12, 64),
                nn.ReLU(True),
                nn.Linear(64, 128),
                nn.ReLU(True),
                nn.Linear(128, image_width * image_height),
                nn.Tanh()
            )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder



