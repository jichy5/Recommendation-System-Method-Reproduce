import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self,sparse_inputs, dense_inputs,labels):
        self.sparse_inputs = sparse_inputs
        self.dense_inputs = dense_inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sparse_inputs[idx],self.dense_inputs[idx],self.labels[idx]


# 生成数据，允许每个特征有不同的类别数量

def generate_data(num_samples=1000, num_sparse_features=3, num_dense_features=3, num_categories_list=None):

    if num_categories_list is None:
        num_categories_list = [10] * num_sparse_features  # 如果没有传入，则默认所有稀疏特征的类别数量相同

    sparse_features = np.zeros((num_samples, num_sparse_features), dtype=int)
    for i in range(num_sparse_features):
        sparse_features[:, i] = np.random.randint(0, num_categories_list[i], size=num_samples)  # 根据不同的类别数量生成稀疏特征

    dense_features = np.random.rand(num_samples, num_dense_features)
    labels = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)

    return torch.tensor(sparse_features, dtype=torch.long), torch.tensor(dense_features,
                                                                         dtype=torch.float32), torch.tensor(labels,
                                                                                                            dtype=torch.float32)