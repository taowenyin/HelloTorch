import faiss
import torch
import torch.nn as nn
import numpy as np


if __name__ == '__main__':
    db_feature = torch.randn(18871, 512)
    q_feature = torch.randn(747, 512)
    n_values = [1, 5, 10, 20, 50, 100]

    batch_size, pooling_dim = db_feature.shape

    # 设置数据的维度
    cph_faiss_index = faiss.IndexFlatL2(pooling_dim)
    # 建立索引
    cph_faiss_index.add(db_feature.cpu().numpy()[:12556, :])
    # 搜索，返回距离和索引
    cph_predictions_distance, cph_predictions_index = cph_faiss_index.search(q_feature.cpu().numpy()[:499, :],
                                                                             max(n_values))


    print('xxx')
