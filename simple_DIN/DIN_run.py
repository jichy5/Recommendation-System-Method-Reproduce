import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_dataloaders
from DIN_class import DIN
from sklearn.metrics import roc_auc_score

def train_and_evaluate():
    embedding_dim = 8
    hidden_units = 64
    dropout_rate = 0.5
    batch_size = 64
    num_epochs = 5
    history_len = 5
    learning_rate = 0.001
    l2_reg = 1e-5

    train_loader, test_loader, feature_info = get_dataloaders(batch_size=batch_size, history_len=history_len)

    # 特征维度信息
    user_num_features = feature_info['user_num_features']
    user_cat_features = feature_info['user_cat_features']
    item_num_features = feature_info['item_num_features']
    item_cat_features = feature_info['item_cat_features']
    encoders = feature_info['encoders']

    feature_sizes = {
        'user_num': len(user_num_features),
        'user_cat': {feat: len(encoders[feat].classes_) for feat in user_cat_features},
        'item_num': len(item_num_features),
        'item_cat': {feat: len(encoders[feat].classes_) for feat in item_cat_features},
        'history_item': 100  # 假设历史物品ID的最大值为100
    }

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DIN(feature_sizes, embedding_dim, hidden_units, dropout_rate)
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    # 训练过程
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for user_num_batch, user_cat_batch, item_num_batch, item_cat_batch, history_batch, label_batch in train_loader:
            user_num_batch = user_num_batch.to(device)
            user_cat_batch = user_cat_batch.to(device)
            item_num_batch = item_num_batch.to(device)
            item_cat_batch = item_cat_batch.to(device)
            history_batch = history_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            outputs = model(user_num_batch, user_cat_batch, item_num_batch, item_cat_batch, history_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # 在测试集上评估
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for user_num_batch, user_cat_batch, item_num_batch, item_cat_batch, history_batch, label_batch in test_loader:
            user_num_batch = user_num_batch.to(device)
            user_cat_batch = user_cat_batch.to(device)
            item_num_batch = item_num_batch.to(device)
            item_cat_batch = item_cat_batch.to(device)
            history_batch = history_batch.to(device)
            label_batch = label_batch.to(device)

            outputs = model(user_num_batch, user_cat_batch, item_num_batch, item_cat_batch, history_batch)
            all_labels.extend(label_batch.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

    auc = roc_auc_score(all_labels, all_preds)
    print(f'Test AUC: {auc:.4f}')

if __name__ == "__main__":
    train_and_evaluate()
