# train_test.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from MF_class import MF  # 引入MF模型

# 1. 生成用户-物品评分矩阵
def generate_user_item_matrix(num_users, num_items, seed=42):
    np.random.seed(seed)
    # 生成 num_users x num_items 的随机评分矩阵，评分范围在 1 到 5 之间
    rating_matrix = np.random.randint(1, 6, size=(num_users, num_items))
    return rating_matrix

# 2. 将评分矩阵转换为用户-物品对的形式 因为MF模型是用这个形式写的
def matrix_to_pairs(rating_matrix):
    user_item_pairs = []
    num_users, num_items = rating_matrix.shape
    for user_id in range(num_users):
        for item_id in range(num_items):
            rating = rating_matrix[user_id, item_id]
            user_item_pairs.append([user_id, item_id, rating])
    return pd.DataFrame(user_item_pairs, columns=['user_id', 'item_id', 'rating'])

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for user_id, item_id, rating in train_loader:
        user_id = user_id.to(device)
        item_id = item_id.to(device)
        rating = rating.to(device).float()

        optimizer.zero_grad()
        outputs = model(user_id,item_id)
        loss = criterion(outputs,rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(train_loader)

def test_model(model, test_loader, device):
    model.eval
    predictions, actuals = [], []
    with torch.no_grad():
        for user_id, item_id, rating in test_loader:
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            rating = rating.to(device).float()

            outputs = model(user_id, item_id)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(rating.cpu().numpy())
    return predictions, actuals



if __name__ == '__main__':
    # 生成用户-物品评分矩阵
    num_users, num_items = 10, 10
    rating_matrix = generate_user_item_matrix(num_users, num_items)

    user_item_df = matrix_to_pairs(rating_matrix)

    train_data, test_data = train_test_split(user_item_df, test_size=0.2, random_state=42)


    def create_data_loader(data, batch_size=32):
        user_ids = torch.tensor(data['user_id'].values, dtype=torch.long)
        item_ids = torch.tensor(data['item_id'].values, dtype=torch.long)
        ratings = torch.tensor(data['rating'].values, dtype=torch.float)
        dataset = torch.utils.data.TensorDataset(user_ids, item_ids, ratings)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_data_loader(train_data)
    test_loader = create_data_loader(test_data)

    # 训练 MF 模型
    embedding_size = 20
    model = MF(num_users, num_items, embedding_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 训练
    epochs = 5
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}')

    # 测试
    predictions, actuals = test_model(model, test_loader, device)
    # RMSE
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # MAE
    mae = mean_absolute_error(actuals, predictions)

    # 打印误差度量
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")