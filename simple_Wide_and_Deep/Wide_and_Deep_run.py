# run.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Wide_and_Deep_class import WideAndDeepModel
from utils import CustomDataset, generate_data,FTRL
from sklearn.model_selection import train_test_split


def train_wide_and_deep_model():
    def evaluate_model(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_sparse_inputs, batch_dense_inputs, batch_labels in test_loader:
                outputs = model(batch_sparse_inputs, batch_dense_inputs).squeeze()
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_labels.squeeze()).sum().item()
                total += batch_labels.size(0)
        print(f"测试准确率: {correct / total:.4f}")

    num_samples = 1000
    num_categories = 4
    embedding_dim = 4
    hidden_units = 16

    # 生成数据
    sparse_inputs, dense_inputs, labels = generate_data(num_samples, num_categories=num_categories)

    # 获取稀疏特征和稠密特征的数量
    num_sparse_features = sparse_inputs.shape[1]
    num_dense_features = dense_inputs.shape[1]

    # 划分数据集
    train_size = int(0.8 * num_samples)
    train_sparse_inputs,test_sparse_inputs, train_dense_inputs, test_dense_inputs,train_labels,test_labels = \
    train_test_split(sparse_inputs, dense_inputs, labels, test_size = 0.2, random_state=32)

    train_dataset = CustomDataset(train_sparse_inputs, train_dense_inputs, train_labels)
    test_dataset = CustomDataset(test_sparse_inputs, test_dense_inputs, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # 初始化模型
    model = WideAndDeepModel(num_categories, embedding_dim, hidden_units, num_sparse_features, num_dense_features)

    # 分离 Wide 和 Deep 部分的参数
    wide_params = []
    deep_params = []
    for name, param in model.named_parameters():
        if 'wide_linear' in name:
            wide_params.append(param)
        else:
            deep_params.append(param)

    # 定义优化器
    # Wide 部分使用带 L1 正则化的 FTRL 优化器
    optimizer_wide = FTRL(wide_params, lr=0.01, l1=1.0, l2=0.0)
    # Deep 部分使用 Adagrad 优化器
    optimizer_deep = optim.Adagrad(deep_params, lr=0.01)

    criterion = nn.BCELoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()

        for batch_sparse_inputs, batch_dense_inputs, batch_labels in train_loader:
            outputs = model(batch_sparse_inputs, batch_dense_inputs).squeeze()
            loss = criterion(outputs, batch_labels.squeeze())

            optimizer_wide.zero_grad()
            optimizer_deep.zero_grad()
            loss.backward()

            optimizer_wide.step()
            optimizer_deep.step()

            epoch_loss += loss.item() * batch_sparse_inputs.size(0)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / train_size:.4f}")

    evaluate_model(model, test_loader)

if __name__ == "__main__":
    train_wide_and_deep_model()
