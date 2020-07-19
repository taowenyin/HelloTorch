import torch
import collections
import utils.tools as tools

from torch import nn

# https://blog.csdn.net/h__ang/article/details/90720579


class AutoEncoderModel(nn.Module):
    '''
    自编码器对象
    input_layer: 输入层神经元数量
    hide_layers: 一半隐藏层神经元数量
    output_layer: 输出层神经元数量
    is_denoising: 是否去噪
    denoising_rate: 去噪比例
    '''
    def __init__(self, input_layer, hide_layers, output_layer,
                 is_denoising = False, denoising_rate = 0):
        super(AutoEncoderModel, self).__init__()

        # 是否添加去噪
        self.is_denoising = is_denoising
        # 去噪比例
        self.denoising_rate = denoising_rate

        # 自编码器的编码层
        encoderLayers = collections.OrderedDict()
        # 创建编码器的隐藏层
        encoder_hide_layers = hide_layers[:(1 if int((len(hide_layers) / 2)) == 0 else int((len(hide_layers) / 2)))]

        # 创建自编码器的编码输入
        encoderLayers['input_layer'] = nn.Linear(input_layer, encoder_hide_layers[0])
        encoderLayers['input_activate'] = nn.ReLU(True)
        # 循环生成编码层
        for i in range(len(encoder_hide_layers)):
            # 判断是否已经到最后一层，没到继续添加，到了就停止
            if i != (len(encoder_hide_layers) - 1):
                hide_layer_label = "encoder_layer_{0}".format(i)
                encoderLayers[hide_layer_label] = nn.Linear(encoder_hide_layers[i],
                                                            encoder_hide_layers[i + 1])
                hide_activate_label = "encoder_activate_{0}".format(i)
                encoderLayers[hide_activate_label] = nn.ReLU(True)
            # 如果隐藏层是奇数，那么编码层还需要和最中间一层链接
            elif (len(hide_layers) % 2 != 0) and (len(hide_layers) > 1):
                hide_layer_label = "encoder_layer_{0}".format(i)
                encoderLayers[hide_layer_label] = nn.Linear(encoder_hide_layers[i], hide_layers[i + 1])
                hide_activate_label = "encoder_activate_{0}".format(i)
                encoderLayers[hide_activate_label] = nn.ReLU(True)
        # 创建自编码器的编码器
        self.encoder = nn.Sequential(encoderLayers)

        # 自编码器的解码层
        decoderLayers = collections.OrderedDict()
        # 判断隐藏层是否为1，如果为1则直接跳到输出层
        if len(hide_layers) > 1:
            # 判断隐藏层是奇数还是偶数
            if len(hide_layers) % 2 == 0:
                # 获取后一半的解码层
                decoder_hide_layers = hide_layers[(len(hide_layers) / 2):]
                # 创建自编码器的解码输入
                decoderLayers['encoder_decoder_layer'] = nn.Linear(encoder_hide_layers[len(encoder_hide_layers) - 1],
                                                                   decoder_hide_layers[0])
            else:
                # 获取后一半的解码层
                decoder_hide_layers = hide_layers[int((len(hide_layers) / 2)) + 1:]
                # 从隐藏层的最终链接到后一层作为解码器的输入
                decoderLayers['encoder_decoder_layer'] = nn.Linear(hide_layers[int((len(hide_layers) / 2))],
                                                                   decoder_hide_layers[0])
            decoderLayers['encoder_decoder_activate'] = nn.ReLU(True)
            # 循环生成解码层
            for i in range(len(decoder_hide_layers)):
                if i != (len(decoder_hide_layers) - 1):
                    hide_layer_label = "decoder_layer_{0}".format(i)
                    decoderLayers[hide_layer_label] = nn.Linear(decoder_hide_layers[i],
                                                                decoder_hide_layers[i + 1])
                    hide_activate_label = "decoder_activate_{0}".format(i)
                    decoderLayers[hide_activate_label] = nn.ReLU(True)
        else:
            decoder_hide_layers = hide_layers

        # 创建自编码器输出
        decoderLayers['output_layer'] = nn.Linear(decoder_hide_layers[len(decoder_hide_layers) - 1], output_layer)
        decoderLayers['output_activate'] = nn.Tanh()
        # 创建自编码器的解码器
        self.decoder = nn.Sequential(decoderLayers)

    def forward(self, x):
        if self.is_denoising:
            # 获取随机噪声
            noise = tools.random_uniform(x.shape, 0, 1 - self.denoising_rate)
            # 获取噪声后的数据
            x = torch.mul(x, noise)
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder
