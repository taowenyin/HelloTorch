# coding: utf-8
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torchvision.transforms as transforms

from PIL import Image
from torchvision.models import mobilenet_v2

# 输入图像路径
single_img_path = r'img/test2.jpg'
# 绘制的热力图存储路径
save_path = r'./img'

# 网络层的层名列表, 需要根据实际使用网络进行修改
layers_names = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7', 'layer8', 'layer9',
                'layer10', 'layer11', 'layer12', 'layer13', 'layer14', 'layer15', 'layer16', 'layer17', 'layer18']
# 指定层名
out_layer_name = "layer18"

features_grad = 0


# 为了读取模型中间参数变量的梯度而定义的辅助函数
def extract(g):
    global features_grad
    features_grad = g


def draw_CAM(model, img_path, save_path, transform=None, out_layer=None):
    """
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :return:
    """
    # 读取图像并预处理
    global layer2
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img).cuda()
    img = img.unsqueeze(0)  # (1, 3, 448, 448)

    # 保存图片的文件名
    save_path = os.path.join(save_path,
                             os.path.basename(img_path).split('.')[0] + '_cam_{}'.format(out_layer) + '.jpg')

    # model转为eval模式
    model.eval()

    # 获取模型层的字典
    layers_dict = {layers_names[i]: None for i in range(len(layers_names))}

    for i, (name, module) in enumerate(model.features._modules.items()):
        layers_dict[layers_names[i]] = module

    # 遍历模型的每一层, 获得指定层的输出特征图
    # features: 指定层输出的特征图, features_flatten: 为继续完成前端传播而设置的变量
    features = img
    start_flatten = False
    features_flatten = None
    for name, layer in layers_dict.items():
        if name != out_layer and start_flatten is False:  # 指定层之前
            features = layer(features)
        elif name == out_layer and start_flatten is False:  # 指定层
            features = layer(features)
            start_flatten = True
        else:  # 指定层之后
            if features_flatten is None:
                features_flatten = layer(features)
            else:
                features_flatten = layer(features_flatten)

    if features_flatten is None:
        features_flatten = features

    # features_flatten = torch.flatten(features_flatten, 1) # SENet的操作
    features_flatten = features_flatten.mean([2, 3]) # MobileNet V2的操作
    output = model.classifier(features_flatten)

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output, 1).item()
    pred_class = output[:, pred]

    # 求中间变量features的梯度
    # 方法1
    # features.register_hook(extract)
    # pred_class.backward()
    # 方法2
    features_grad = autograd.grad(pred_class, features, allow_unused=True)[0]

    grads = features_grad  # 获取梯度
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    print("pooled_grads:", pooled_grads.shape)
    print("features:", features.shape)
    # features.shape[0]是指定层feature的通道数
    for i in range(features.shape[0]):
        features[i, ...] *= pooled_grads[i, ...]

    # 计算heatmap
    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 保存原始的热力图
    org_heatmap = heatmap.copy()

    # 设置热力图
    img = Image.open(img_path)  # 用Image加载原始图像
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式

    plt.imshow(img, alpha=1)
    plt.imshow(heatmap, alpha=0.6, interpolation='nearest', cmap='jet')
    plt.colorbar()
    plt.savefig(save_path, dpi=100)
    plt.show()


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # 构建模型并加载预训练参数
    net = mobilenet_v2(pretrained=True).cuda()

    for i, name in enumerate(layers_names):
        if i == 0:
            continue
        draw_CAM(net, single_img_path, save_path, transform=transform, out_layer=name)

