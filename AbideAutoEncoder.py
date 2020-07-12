"""

Autoencoders training and fine-tuning.

Usage:
  nn.py [--whole] [--male] [--threshold] [--leave-site-out] [<derivative> ...]
  nn.py (-h | --help)

Options:
  -h --help           Show this screen
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  derivative          Derivatives to process

"""

import torch

from docopt import docopt
from utils.data.MnistData import MnistData
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn, optim
from model.AutoEncoderModel import AutoEncoderModel

import utils.abide.prepare_utils as PrepareUtils

if __name__ == '__main__':
    # 模型初始化
    PrepareUtils.reset()

    arguments = docopt(__doc__)

    # 表型数据位置
    pheno_path = './data/ABIDE/phenotypes/Phenotypic_V1_0b_preprocessed1.csv'
    # 载入表型数据
    pheno = PrepareUtils.load_phenotypes(pheno_path)
    # 载入数据集
    hdf5 = PrepareUtils.hdf5_handler(bytes('./data/ABIDE/abide.hdf5', encoding='utf8'), 'a')

    # 脑图谱的选择
    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative
                   in arguments["<derivative>"]
                   if derivative in valid_derivatives]

    # 标记实现数据
    experiments = []
    for derivative in derivatives:

        config = {"derivative": derivative}

        if arguments["--whole"]:
            experiments += [PrepareUtils.format_config("{derivative}_whole", config)],

        if arguments["--male"]:
            experiments += [PrepareUtils.format_config("{derivative}_male", config)]

        if arguments["--threshold"]:
            experiments += [PrepareUtils.format_config("{derivative}_threshold", config)]

        if arguments["--leave-site-out"]:
            for site in pheno["SITE_ID"].unique():
                site_config = {"site": site}
                experiments += [
                    PrepareUtils.format_config("{derivative}_leavesiteout-{site}",
                                  config, site_config)
                ]

    # 第一个自编码器的隐藏层神经元数量
    code_size_1 = 1000
    # 第二个自编码器的隐藏层神经元数量
    code_size_2 = 600

    # 要训练的脑图谱列表排序
    experiments = sorted(experiments)
    # 循环训练所有脑图谱
    for experiment_item in experiments:
        # 获得脑图谱名称
        experiment = experiment_item[0]
        # 从HDF5载入实验数据
        exp_storage = hdf5["experiments"][experiment]

        for fold in exp_storage:
            experiment_cv = PrepareUtils.format_config("{experiment}_{fold}", {
                "experiment": experiment,
                "fold": fold,
            })
            # 获取训练数据、验证数据、测试数据
            X_train, y_train, X_valid, y_valid, X_test, y_test = PrepareUtils.load_fold(hdf5["patients"], exp_storage, fold)

            # 保存AE1模型的地址
            ae1_model_path = PrepareUtils.format_config("./data/ABIDE/models/{experiment}_autoencoder-1.ckpt", {
                "experiment": experiment_cv,
            })
            # 保存AE2模型的地址
            ae2_model_path = PrepareUtils.format_config("./data/ABIDE/models/{experiment}_autoencoder-2.ckpt", {
                "experiment": experiment_cv,
            })
            # 保存NN模型的地址
            nn_model_path = PrepareUtils.format_config("./data/ABIDE/models/{experiment}_mlp.ckpt", {
                "experiment": experiment_cv,
            })

        # PrepareUtils.run_nn(hdf5, experiment[0], code_size_1, code_size_2)

    a = AutoEncoderModel(1000, [50, 20, 30], 700)
    print('xxx')

    # # 每批数据的大小
    # batch_size = 100
    #
    # # 创建数据
    # dataset = MnistData()
    # train_labels, train_data = dataset.parse_train_data()
    # test_labels, test_data = dataset.parse_test_target()
    # validation_labels, validation_data = dataset.parse_validation_data()
    #
    # # 创建训练数据集
    # train_dataset = TensorDataset(
    #     train_data.float().clone().detach().requires_grad_(True),
    #     train_labels.float().clone().detach().requires_grad_(True))
    # # 创建训练数据集
    # test_dataset = TensorDataset(
    #     test_data.float().clone().detach().requires_grad_(True),
    #     test_labels.float().clone().detach().requires_grad_(True))
    # # 创建验证数据集
    # validation_dataset = TensorDataset(
    #     validation_data.float().clone().detach().requires_grad_(True),
    #     validation_labels.float().clone().detach().requires_grad_(True))
    #
    # # 创建训练数据加载器，并且设置每批数据的大小，以及每次读取数据时随机打乱数据
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # # 创建验证集加载器
    # validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    # # 测试集加载器
    # test_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)
    #
    # # 创建模型
    # model = AutoEncoderModel(28, 28, AutoEncoderType.Multi)
    # # 均方差损失函数
    # criterion = nn.MSELoss()
    # # 采用Adam优化
    # optimizier = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)