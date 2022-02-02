import torch.nn as nn
import torch.optim as optim

from torchvision.models import vgg16


if __name__ == '__main__':
    # 创建预训练模型
    net = vgg16(pretrained=True)
    layers = list(net.features.children())

    print('======冻结网络参数学习======')
    for l, value in enumerate(zip(net.parameters(), layers)):
        # 获取网络层的参数
        param = value[0]
        # 获取网络层
        layer = value[1]
        # 冻结网络层的参数学习，即停止梯度下降
        param.requires_grad = False

        print('======Layout {}======'.format(l))
        print(layer)
        print('Param Shape = {}'.format(param.shape))

    print('======修改某一层网络，修改的网络可以进行参数学习======')
    net_layers = list(net.features.children())
    net_layers[-1] = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    net = nn.Sequential(*net_layers)
    for l, param in enumerate(net.parameters()):
        # 查看新增加的网络是否可以梯度下降
        print('======Layout {}, requires_grad = {}. shape = {}======'.format(l, param.requires_grad, param.shape))

    print('======针对某些层进行全局调整======')
    net = vgg16(pretrained=True)
    # 筛选出要进行单独设置的网络层参数
    ignored_params = list(map(id, net.features[0].parameters()))
    # 获取其他网络层参数
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    # 整个网络层的参数为后面部分，单独设置的为括号中
    optimizer = optim.SGD([
        {'params': base_params},
        {'params': net.features[0].parameters(), 'lr': 0.001 * 10}
    ], lr=0.001, momentum=0.9, weight_decay=1e-4)
