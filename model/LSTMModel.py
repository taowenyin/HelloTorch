import torch

from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 创建模型
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 创建全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        # 设置激活函数
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_x, hidden):
        output, (hidden_n, cell_n) = self.lstm(input_x, hidden)
        output = output[:, -1, :]
        output = self.fc(output)
        output = self.softmax(output)
        return output, (hidden_n, cell_n)

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        cell = torch.zeros(self.num_layers, 1, self.hidden_size)

        return hidden, cell
