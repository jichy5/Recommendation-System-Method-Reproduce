# utils.py

import torch
from torch.utils.data import Dataset
import numpy as np
from torch.optim import Optimizer

class CustomDataset(Dataset):
    def __init__(self, sparse_inputs, dense_inputs, labels):
        self.sparse_inputs = sparse_inputs
        self.dense_inputs = dense_inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sparse_inputs[idx], self.dense_inputs[idx], self.labels[idx]

def generate_data(num_samples=1000, num_categories=4):
    num_sparse_features = np.random.randint(2, 5)  # 随机稀疏特征数量（2到4个）
    num_dense_features = np.random.randint(2, 5)   # 随机稠密特征数量（2到4个）
    sparse_features = np.random.randint(0, num_categories, size=(num_samples, num_sparse_features))
    dense_features = np.random.rand(num_samples, num_dense_features)
    labels = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)

    return torch.tensor(sparse_features, dtype=torch.long), \
           torch.tensor(dense_features, dtype=torch.float32), \
           torch.tensor(labels, dtype=torch.float32)


class FTRL(Optimizer):
    def __init__(self, params, lr=1.0, l1=1.0, l2=1.0, beta=1.0):
        defaults = dict(lr=lr, l1=l1, l2=l2, beta=beta)
        super(FTRL, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            l1 = group['l1']
            l2 = group['l2']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['z'] = torch.zeros_like(p.data)
                    state['n'] = torch.zeros_like(p.data)

                z = state['z']
                n = state['n']

                n += grad.pow(2)
                sigma = (n - torch.sqrt(n)) / lr
                z += grad - sigma * p.data

                # Proximal update with L1 regularization
                p.data = (torch.sign(z) * torch.clamp(torch.abs(z) - l1, min=0)) / (l2 + (beta + torch.sqrt(n)) / lr)

        return loss