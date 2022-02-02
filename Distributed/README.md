# 分布式训练

[DistributedDataParallel（DDP）和Horovod的简明使用教程。](https://zhuanlan.zhihu.com/p/170512930)

1. test()函数没必要在主GPU之外的GPU上运行，产生多个输出没必要啊，当然你一定要运行也是没关系的。就是同时往外存吐的时候可能会冲突。 
2. tensorboard存数据的时候,或者存模型的时候，记得只在rank0上使用，加一句 if rank == 0:
3. torch.load()的时候一定不要忘了map_location的操作，一定啊！！ 
4. 对待GPU不搞极限施压，显存用到85%左右就行了，偶尔有些操作会往上窜一窜。一点余量都不给别怪它最后OOM了。Memory-Usage用多少不是关键，GPU-Util拉满才是王道，毕竟后者才是实打实的计算。
