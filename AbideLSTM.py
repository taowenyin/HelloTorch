"""

LSTM training and fine-tuning.

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


import numpy as np
import pandas as pd
import utils.abide.prepare_utils as PrepareUtils

from docopt import docopt


if __name__ == '__main__':
    arguments = docopt(__doc__)

    # 表型数据位置
    pheno_path = './data/ABIDE/phenotypes/Phenotypic_V1_0b_preprocessed1.csv'
    # 载入表型数据
    pheno = PrepareUtils.load_phenotypes(pheno_path)
    # 载入数据集
    hdf5 = PrepareUtils.hdf5_handler(bytes('./data/ABIDE/abide_lstm.hdf5', encoding='utf8'), 'a')

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

    # 要训练的脑图谱列表排序
    experiments = sorted(experiments)
    # 循环训练所有脑图谱
    for experiment_item in experiments:
        # 获得脑图谱名称
        experiment = experiment_item[0]
        # 从HDF5载入实验数据
        exp_storage = hdf5["experiments"][experiment]

        # 循环获得每折数据
        for fold in exp_storage:
            experiment_cv = PrepareUtils.format_config("{experiment}_{fold}", {
                "experiment": experiment,
                "fold": fold,
            })

            # 获取训练数据、验证数据、测试数据
            X_train, y_train, X_valid, y_valid, X_test, y_test = PrepareUtils.load_fold(
                hdf5["patients"], exp_storage, fold)

            print('xxx')

        print('xxx')

    print('xxx')