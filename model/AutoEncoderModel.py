from torch import nn

# https://blog.csdn.net/h__ang/article/details/90720579

class AutoEncoderModel(nn.Module):
    '''
    自编码器对象
    '''
    def __init__(self):
        super(AutoEncoderModel, self).__init__()