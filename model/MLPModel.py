import torch

from torch import nn


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()