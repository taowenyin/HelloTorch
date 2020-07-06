from typing import Any

from torch import nn

# https://blog.csdn.net/h__ang/article/details/90720579
from torch.nn.modules.module import T_co


class AutoEncoderModel(nn.Module):
    '''
    自编码器对象
    '''
    def __init__(self, image_width, image_height):
        super(AutoEncoderModel, self).__init__()
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



