import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        #
        # embed_dim表示切好的图片拉成一维向量后的特征长度
        #
        # 图像共切分为N = HW/P^2个patch块
        # 在实现上等同于对reshape后的patch序列进行一个PxP且stride为P的卷积操作
        # output = {[(n+2p-f)/s + 1]向下取整}^2
        # 即output = {[(n-P)/P + 1]向下取整}^2 = (n/P)^2
        #
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x_1 = self.proj(x)
        x_2 = x_1.flatten(2)
        out = x_2.transpose(1, 2)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        return out  # x.shape is [8, 196, 768]


if __name__ == '__main__':
    input = torch.rand(1, 3, 224, 224)
    model = PatchEmbed()

    out = model(input)
    print(out.shape)

