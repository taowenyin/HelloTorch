import torch
import collections

from torch import nn
from enum import Enum

# https://blog.csdn.net/h__ang/article/details/90720579


class AutoEncoderModel(nn.Module):
    '''
    自编码器对象
    input_layer: 输入层神经元数量
    half_hide_layers: 一半隐藏层神经元数量
    output_layer: 输出层神经元数量
    learning_rate: 学习率
    is_sparse: 是否稀疏
    sparse_param: 稀疏参数
    sparse_coeff: 稀疏系数
    is_denoising: 是否去噪
    denoising_rate: 去噪比例
    '''
    def __init__(self, input_layer, half_hide_layers, output_layer,
                 learning_rate = 0, is_sparse = False, sparse_param = 0, sparse_coeff = 0,
                 is_denoising = False, denoising_rate = 0):
        super(AutoEncoderModel, self).__init__()

        # 自编码器的编码层
        encoderLayers = collections.OrderedDict()
        # 创建自编码器输入
        encoderLayers['input_layer'] = nn.Linear(input_layer, half_hide_layers[0])
        encoderLayers['input_activate'] = nn.ReLU(True)
        # 循环生成编码层
        for i in range(len(half_hide_layers)):
            if i != (len(half_hide_layers) - 1):
                hide_layer_label = "encoder_layer_{0}".format(i)
                encoderLayers[hide_layer_label] = nn.Linear(half_hide_layers[i], half_hide_layers[i + 1])
                hide_activate_label = "encoder_activate_{0}".format(i)
                encoderLayers[hide_activate_label] = nn.ReLU(True)

        # 创建自编码器的编码器
        self.encoder = nn.Sequential(encoderLayers)

        # 隐藏顺序去反
        half_hide_layers.reverse()

        # 自编码器的解码层
        decoderLayers = collections.OrderedDict()
        # 循环生成解码层
        for i in range(len(half_hide_layers)):
            if i != (len(half_hide_layers) - 1):
                hide_layer_label = "decoder_layer_{0}".format(i)
                decoderLayers[hide_layer_label] = nn.Linear(half_hide_layers[i], half_hide_layers[i + 1])
                hide_activate_label = "decoder_activate_{0}".format(i)
                decoderLayers[hide_activate_label] = nn.ReLU(True)
        # 创建自编码器输出
        decoderLayers['output_layer'] = nn.Linear(half_hide_layers[len(half_hide_layers) - 1], output_layer)
        decoderLayers['output_activate'] = nn.Tanh()

        # 创建自编码器的解码器
        self.decoder = nn.Sequential(decoderLayers)

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder



