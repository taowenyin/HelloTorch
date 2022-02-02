import torch
import os
import tempfile
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


# 分布式模型，把模型计算放在不同GPU上
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)


def setup(rank, world_size):
    """
    分布式训练初始化

    Parameters
    ----------
    rank: 当前进程中GPU的编号
    world_size: 总共有多少个GPU
    """
    # 确定可用的GPU，注意这句话一定要放在任何对CUDA的操作之前（和别人公用服务器时使用）
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # 设置两个环境变量，localhost是本地的ip地址，12355是用来通信的端口
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    # 实现GPU的负载均衡
    torch.cuda.set_device(rank)


def cleanup():
    """
    在所有任务行完以后消灭进程用的。
    """

    dist.destroy_process_group()


def run_parallel(parallel_fn, world_size):
    """
    多进程产生函数。不用这个的话需要在运行训练代码的时候，用
    'python-m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train.py'才能启动。

    Parameters
    ----------
    parallel_fn: 分布式的函数
    world_size: GPU数量

    Returns
    -------

    """
    mp.spawn(parallel_fn, args=(world_size,), nprocs=world_size, join=True)


def main_train_basic(rank, world_size):
    """
    基本的训练模板，需要指定device_ids让模型运行在不同的GPU上

    Parameters
    ----------
    rank: 当前进程中GPU的编号
    world_size: 总共有多少个GPU

    """
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def main_train_basic_plus(rank, world_size):
    """
    带Checkpoint保存和读取的训练过程，需要指定device_ids让模型运行在不同的GPU上

    Parameters
    ----------
    rank: 当前进程中GPU的编号
    world_size: 总共有多少个GPU

    """
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    # 创建模型，并把它移动到对应的GPU上
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"

    # 保存Checkpoint
    if rank == 0:
        # 因为每个GPU上的参数都是相同的，因此只需要保存一个GPU上的参数就行
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # 等待GPU:0保存参数完毕
    dist.barrier()

    # GPU的位置映射，torch.load一定要加
    # 如果少了这句话，就会主GPU干活，剩下GPU在看戏的问题
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=map_location)
    ddp_model.load_state_dict(checkpoint)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # 再删除文件时不需要使用dist.barrier()来保护同步，因为DDP反向传播时所有的AllReduce ops都已经同步
    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


def main_train_basic_parallel(rank, world_size):
    """
    带分布式模型的的训练过程，不需要指定device_ids，因为在哪个设备上运行已经在模型中声明

    Parameters
    ----------
    rank
    world_size
    """

    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # 设置模型和设备
    dev0 = (rank * 2) % world_size
    dev1 = (rank * 2 + 1) % world_size
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()

    # 从dev1返回数据
    outputs = ddp_mp_model(torch.randn(20, 10))
    # 创建dev1的Label数据
    labels = torch.randn(20, 5).to(dev1)
    # 在dev1上计算损失
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


if __name__ == '__main__':
    # # 告诉每个GPU各自使用哪些数据
    # trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # trainloader = DataLoader(一堆你自己的的参数, sampler=trainsampler)

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"至少需要2个GPUs，现在有{n_gpus}个GPUs"
    world_size = n_gpus

    # run_parallel(main_train_basic, world_size)
    # run_parallel(main_train_basic_plus, world_size)
    run_parallel(main_train_basic_parallel, world_size)
