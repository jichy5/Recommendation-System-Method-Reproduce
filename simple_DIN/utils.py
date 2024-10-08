import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 数据生成函数
def generate_synthetic_data(num_samples=1000, history_len=5):
    """
    生成模拟的用户、物品、历史行为和标签数据。
    """
    # 定义特征维度
    user_num_features = ['age', 'income']
    user_cat_features = ['gender', 'occupation']
    item_num_features = ['price', 'discount']
    item_cat_features = ['category', 'brand']

    # 数值特征生成
    user_num_data = {
        'age': np.random.randint(18, 65, size=num_samples),
        'income': np.random.uniform(2000, 10000, size=num_samples)
    }
    item_num_data = {
        'price': np.random.uniform(10, 500, size=num_samples),
        'discount': np.random.uniform(0, 0.5, size=num_samples)
    }

    # 类别特征生成
    user_cat_data = {
        'gender': np.random.choice(['M', 'F'], size=num_samples),
        'occupation': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Lawyer'], size=num_samples)
    }
    item_cat_data = {
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], size=num_samples),
        'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC'], size=num_samples)
    }

    # 历史行为（物品ID）
    history_items = np.random.randint(0, 100, size=(num_samples, history_len))

    # 标签
    labels = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32)

    # 合并所有数据
    user_data = {**user_num_data, **user_cat_data}
    item_data = {**item_num_data, **item_cat_data}

    # 创建 DataFrame
    user_df = pd.DataFrame(user_data)
    item_df = pd.DataFrame(item_data)
    history_df = pd.DataFrame({'history': list(history_items)})
    label_df = pd.DataFrame({'label': labels})

    # 合并成一个 DataFrame
    data = pd.concat([user_df, item_df, history_df, label_df], axis=1)

    return data, user_num_features, user_cat_features, item_num_features, item_cat_features

# 数据预处理函数
def preprocess_data(data, user_num_features, user_cat_features, item_num_features, item_cat_features):
    """
    对数据进行预处理，包括数值特征归一化、类别特征编码等。
    """
    # 数值特征归一化
    scaler = MinMaxScaler()
    data[user_num_features + item_num_features] = scaler.fit_transform(data[user_num_features + item_num_features])

    # 类别特征编码
    encoders = {}
    for feature in user_cat_features + item_cat_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        encoders[feature] = le

    return data, encoders

# 自定义 Dataset
class DINDataSet(Dataset):
    def __init__(self, data, user_num_features, user_cat_features, item_num_features, item_cat_features):
        self.user_num = data[user_num_features].values.astype(np.float32)
        self.user_cat = data[user_cat_features].values.astype(np.int64)
        self.item_num = data[item_num_features].values.astype(np.float32)
        self.item_cat = data[item_cat_features].values.astype(np.int64)
        self.history = np.stack(data['history'].values)
        self.labels = data['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_num = torch.tensor(self.user_num[idx])
        user_cat = torch.tensor(self.user_cat[idx])
        item_num = torch.tensor(self.item_num[idx])
        item_cat = torch.tensor(self.item_cat[idx])
        history = torch.tensor(self.history[idx])
        label = torch.tensor(self.labels[idx])

        return user_num, user_cat, item_num, item_cat, history, label

# 创建数据加载器
def get_dataloaders(batch_size=64, history_len=5, test_size=0.2):
    data, user_num_features, user_cat_features, item_num_features, item_cat_features = generate_synthetic_data(history_len=history_len)
    data, encoders = preprocess_data(data, user_num_features, user_cat_features, item_num_features, item_cat_features)

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    train_dataset = DINDataSet(train_data, user_num_features, user_cat_features, item_num_features, item_cat_features)
    test_dataset = DINDataSet(test_data, user_num_features, user_cat_features, item_num_features, item_cat_features)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    feature_info = {
        'user_num_features': user_num_features,
        'user_cat_features': user_cat_features,
        'item_num_features': item_num_features,
        'item_cat_features': item_cat_features,
        'history_len': history_len,
        'encoders': encoders
    }

    return train_loader, test_loader, feature_info
