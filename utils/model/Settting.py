import torch
import numpy as np
import random


# 初始化随机种子
def reset_torch():
    random.seed(19)
    torch.manual_seed(19)
    np.random.seed(19)