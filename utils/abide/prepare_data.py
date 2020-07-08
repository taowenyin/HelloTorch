"""

Data preparation

Usage:
  prepare_data.py [--folds=N] [--whole] [--male] [--threshold] [--leave-site-out] [<derivative> ...]
  prepare_data.py (-h | --help)

Options:
  -h --help           Show this screen
  --folds=N           Number of folds [default: 10]
  --whole             Prepare data of the whole dataset
  --male              Prepare data of male subjects
  --threshold         Prepare data of thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  derivative          Derivatives to process

"""


import os
import random
import pandas as pd
import numpy as np
import numpy.ma as ma
from docopt import docopt
from functools import partial
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils.abide.prepare_utils import (load_phenotypes, format_config, run_progress, hdf5_handler)


# 计算每个ROI的连接性
def compute_connectivity(functional):
    with np.errstate(invalid="ignore"):
        # 计算每个ROI的相关性
        corr = np.nan_to_num(np.corrcoef(functional))
        # 相关性矩阵的一半可以去掉
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        # 获得蒙版
        m = ma.masked_where(mask == 1, mask)
        # 根据蒙版获取数据
        return ma.masked_where(m, corr).compressed()


def load_patient(subj, tmpl):
    # 拼接参数获取数据地址，并读取数据
    df = pd.read_csv(format_config(tmpl, {
        "subject": subj,
    }), sep="\t", header=0)
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # 获取ROI区域编号
    ROIs = ["#" + str(y) for y in sorted([int(x[1:]) for x in df.keys().tolist()])]
    # 使用0替代无效元素，一共200行，表示200个感兴趣区域，每行一共有196个元素，表示每个感兴趣区域有196个值
    functional = np.nan_to_num(df[ROIs].to_numpy().T).tolist()
    # axis=1表示沿着x轴数据标准化
    functional = preprocessing.scale(functional, axis=1)
    # 计算并获得每两个ROI之间的连接性
    functional = compute_connectivity(functional)
    functional = functional.astype(np.float32)
    # 返回某个病人的ROI的连接性
    return subj, functional.tolist()


# 启动线程载入患者数据
def load_patients(subjs, tmpl, jobs=1):
    # 构建函数对象
    partial_load_patient = partial(load_patient, tmpl=tmpl)
    msg = "Processing {current} of {total}"
    # 启动线程载入患者数据，并返回字典数据
    return dict(run_progress(partial_load_patient, subjs, message=msg, jobs=jobs))


def prepare_folds(hdf5, folds, pheno, derivatives, experiment):
    exps = hdf5.require_group("experiments")
    ids = pheno["FILE_ID"]

    for derivative in derivatives:
        exp = exps.require_group(format_config(
            experiment,
            {
                "derivative": derivative,
            }
        ))

        exp.attrs["derivative"] = derivative

        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        for i, (train_index, test_index) in enumerate(skf.split(ids, pheno["STRAT"])):
            train_index, valid_index = train_test_split(train_index, test_size=0.33)
            fold = exp.require_group(str(i))

            fold['train'] = [ind.encode('utf8') for ind in ids[train_index]] 

            fold['valid'] = [indv.encode('utf8') for indv in ids[valid_index]]
 
            fold["test"] = [indt.encode('utf8') for indt in ids[test_index]]

            # fold["train"] = ids[train_index].tolist()
            # fold["valid"] = ids[valid_index].tolist()
            # fold["test"] = ids[test_index].tolist()


# 把患者的数据保存到HDF5中
def load_patients_to_file(hdf5, pheno, derivatives):

    download_root = "../../data/ABIDE/functionals"
    derivatives_path = {
        "aal": "cpac/filt_global/rois_aal/{subject}_rois_aal.1D",
        "cc200": "cpac/filt_global/rois_cc200/{subject}_rois_cc200.1D",
        "dosenbach160": "cpac/filt_global/rois_dosenbach160/{subject}_rois_dosenbach160.1D",
        "ez": "cpac/filt_global/rois_ez/{subject}_rois_ez.1D",
        "ho": "cpac/filt_global/rois_ho/{subject}_rois_ho.1D",
        "tt": "cpac/filt_global/rois_tt/{subject}_rois_tt.1D",
    }
    # 创建患者分组，并返回对象
    storage = hdf5.require_group("patients")
    # 获取患者的文件ID
    file_ids = pheno["FILE_ID"].tolist()

    # 遍历所有的脑图谱
    for derivative in derivatives:
        # 拼接脑图谱文件
        file_template = os.path.join(download_root, derivatives_path[derivative])
        # 把患者数据文件转化为每个病人的每个ROI连接性数据
        func_data = load_patients(file_ids, tmpl=file_template)

        for pid in func_data:
            print('func_data_filling')
            record = pheno[pheno["FILE_ID"] == pid].iloc[0]
            # 每个病人创建一个分组
            patient_storage = storage.require_group(pid)
            # 保存每个病人的个人数据
            patient_storage.attrs["id"] = record["FILE_ID"]
            patient_storage.attrs["y"] = record["DX_GROUP"]
            patient_storage.attrs["site"] = record["SITE_ID"]
            patient_storage.attrs["sex"] = record["SEX"]
            # 保存每个病人的数据
            patient_storage.create_dataset(derivative, data=func_data[pid])

if __name__ == "__main__":

    random.seed(19)
    np.random.seed(19)

    arguments = docopt(__doc__)

    folds = int(arguments["--folds"])
    # 表型数据的地址
    pheno_path = "../../data/ABIDE/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    # 读取表型数据
    pheno = load_phenotypes(pheno_path)

    # 创建HDF5数据文件
    hdf5 = hdf5_handler(bytes("../../data/ABIDE/abide.hdf5", encoding="utf8"), 'a')

    # 不同的脑图谱
    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative in arguments["<derivative>"] if derivative in valid_derivatives]

    #  如果没有患者数据，那么就要读取患者的ROI的连接性数据并把数据保存到HDF5中
    if "patients" not in hdf5:
        load_patients_to_file(hdf5, pheno, derivatives)

    if arguments["--whole"]:
        print ("Preparing whole dataset")
        prepare_folds(hdf5, folds, pheno, derivatives, experiment="{derivative}_whole")

    if arguments["--male"]:
        
        print ("Preparing male dataset")
        pheno_male = pheno[pheno["SEX"] == "M"]
        prepare_folds(hdf5, folds, pheno_male, derivatives, experiment="{derivative}_male")

    if arguments["--threshold"]:
        
        print ("Preparing thresholded dataset")
        pheno_thresh = pheno[pheno["MEAN_FD"] <= 0.2]
        prepare_folds(hdf5, folds, pheno_thresh, derivatives, experiment="{derivative}_threshold")

    if arguments["--leave-site-out"]:
        
        # print('Hi')
        print ("Preparing leave-site-out dataset")
        for site in pheno["SITE_ID"].unique():
            pheno_without_site = pheno[pheno["SITE_ID"] != site]
            prepare_folds(hdf5, folds, pheno_without_site, derivatives, experiment=format_config(
                "{derivative}_leavesiteout-{site}",
                {
                    "site": site,
                })
            )
