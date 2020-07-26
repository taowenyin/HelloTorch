import torch
import collections

from torch import nn


class MLPModel(nn.Module):
    def __init__(self, input_layer, hide_layers, output_layer, dropout_layers):
        """
        多层感知机
        :param input_layer: 输入层特征数
        :param hide_layers: 隐藏层特征数
        :param output_layer: 输出层特征数
        :param drop_layers: Dropout层
        """
        super(MLPModel, self).__init__()

        self.mlp_layers = collections.OrderedDict()

        for i in range(len(hide_layers)):
            if i == 0:
                self.mlp_layers['layer_{0}'.format(i)] = nn.Linear(input_layer, hide_layers[i])
            else:
                self.mlp_layers['layer_{0}'.format(i)] = nn.Linear(hide_layers[i - 1], hide_layers[i])
            # 添加Dropout层
            self.mlp_layers['layer_{0}_dropout'.format(i)] = nn.Dropout(dropout_layers[i])
            # 添加激活层
            self.mlp_layers['layer_{0}_act'.format(i)] = nn.Tanh()

        self.mlp_layers['layer_{0}'.format(len(hide_layers))] = nn.Linear(hide_layers[len(hide_layers) - 1], output_layer)
        # self.mlp_layers['layer_{0}_act'.format(len(hide_layers))] = nn.Softmax()

        self.layers = nn.Sequential(self.mlp_layers)

    def forward(self, x):
        return self.layers(x)
