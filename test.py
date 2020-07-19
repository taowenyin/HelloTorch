import numpy as np
import torch


if __name__ == '__main__':
    x = torch.arange(12).view(4, 3).float()
    x_mean0 = torch.mean(x, dim=0, keepdim=True)
    x_mean0_keep = torch.mean(x, dim=0)

    print(x)
    print(x_mean0)
    print(x_mean0_keep)