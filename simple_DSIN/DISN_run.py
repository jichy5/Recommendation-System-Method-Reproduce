import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from utils import generate_hist_sess, preprocess_data, prepare_input
from DISN_class import DSIN


np.random.seed(42)
torch.manual_seed(42)

# 生成模拟数据
num_samples = 1000  # 样本数量
max_sess_count = 3  # 最大会话数量
max_seq_len = 5     # 每个会话的最大序列长度

# 构建 DataFrame，包含用户特征、密集特征和标签
samples_data = pd.DataFrame({
    'user_id': np.random.randint(0, 5, size=num_samples),
    'gender': np.random.randint(0, 2, size=num_samples),
    'age': np.random.randint(18, 60, size=num_samples),
    'movie_id': np.random.randint(1, 20, size=num_samples),
    'movie_type_id': np.random.randint(1, 5, size=num_samples),
    'occupation': np.random.randint(0, 10, size=num_samples),
    'salary': np.random.rand(num_samples) * 10000,  # 密集特征：工资
    'watch_count': np.random.randint(1, 100, size=num_samples),  # 密集特征：观看次数
    'label': np.random.randint(0, 2, size=num_samples)
})

# 生成历史会话数据
sess_array_list = []
sess_lengths_list = []
sess_length_list = []
for _ in range(num_samples):
    sess_array, sess_lengths, sess_length = generate_hist_sess(max_sess_count, max_seq_len)
    sess_array_list.append(sess_array)
    sess_lengths_list.append(sess_lengths)
    sess_length_list.append(sess_length)

samples_data['sess_array'] = sess_array_list
samples_data['sess_lengths'] = sess_lengths_list
samples_data['sess_length'] = sess_length_list
print(samples_data['sess_array'])
# 定义特征
sparse_features = ['user_id', 'gender', 'age', 'movie_id', 'movie_type_id', 'occupation']
dense_features = ['salary', 'watch_count']  # 添加密集特征
varlen_sparse_features = ['sess_array']

# 数据预处理
samples_data, feature_columns = preprocess_data(samples_data, sparse_features, dense_features, varlen_sparse_features, max_seq_len)
print(feature_columns)
# 行为特征列表 重点的关注
sess_feature_list = ['movie_id']

# 划分训练集和测试集
train_data, test_data = train_test_split(samples_data, test_size=0.2, random_state=42)

# 准备输入数据
X_train, y_train = prepare_input(train_data, sparse_features, dense_features, varlen_sparse_features)
X_test, y_test = prepare_input(test_data, sparse_features, dense_features, varlen_sparse_features)

# 定义PyTorch数据集
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {}
        for key in self.X:
            sample[key] = torch.tensor(self.X[key][idx])
        label = torch.tensor(self.y[idx], dtype=torch.float)
        return sample, label


train_dataset = SimpleDataset(X_train, y_train)
test_dataset = SimpleDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 初始化并训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DSIN(feature_columns, sess_feature_list, bias_encoding=True)
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

epochs = 5
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        for key in batch_X:
            batch_X[key] = batch_X[key].to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs,batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# 测试
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        for key in batch_X:
            batch_X[key] = batch_X[key].to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_X)
        preds = outputs.cpu().numpy()
        labels = batch_y.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

print(all_preds)
# 计算Accuracy和AUC
threshold = 0.5
binary_preds = [1 if x >= threshold else 0 for x in all_preds]
accuracy = accuracy_score(all_labels, binary_preds)
auc = roc_auc_score(all_labels, all_preds)

print(f"Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
