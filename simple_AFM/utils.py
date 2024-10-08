import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, sparse_inputs, dense_inputs, labels):
        self.sparse_inputs = sparse_inputs
        self.dense_inputs = dense_inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sparse_inputs[idx],self.dense_inputs[idx],self.labels[idx]

def generate_data(nums_sample = 1000, nums_sparse = 3, nums_dense = 3, categories_list = None):
    if categories_list == None:
        categories_list = [10] * nums_sparse

    sparse_feature = np.zeros((nums_sample, nums_sparse),dtype = int)
    for i in range(nums_sparse):
        sparse_feature[:,i] = np.random.randint(0,categories_list[i],size = nums_sample)

    dense_feature = np.random.rand(nums_sample,nums_dense)
    labels = np.random.randint(0,2,size=(nums_sample,1)).astype(np.float32)

    return torch.tensor(sparse_feature,dtype=torch.long), torch.tensor(dense_feature,dtype=torch.float32),\
            torch.tensor(labels,dtype=torch.float32)
