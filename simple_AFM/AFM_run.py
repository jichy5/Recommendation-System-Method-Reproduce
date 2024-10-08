import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from AFM_class import AFM
from utils import CustomDataset,generate_data

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_sparse_inputs, batch_dense_inputs, batch_labels in test_loader:
            output = model(batch_sparse_inputs, batch_dense_inputs).squeeze()
            prediction = (output > 0.5).float()
            correct += (prediction == batch_labels.squeeze()).sum().item()
            total += batch_labels.size(0)
    print(f'Test Accuracy: {correct / total:.4f}')

class train_afm_model():
    num_samples = 1000
    num_sparse_features = 3
    num_dense_features = 3
    num_categories_list = [5, 10, 15]
    embedding_dim = 8
    hidden_units = 64
    att_embedding_dim = 4

    sparse_inputs, dense_inputs,labels = generate_data(num_samples,num_sparse_features,num_dense_features,num_categories_list)

    train_sparse_inputs, test_sparse_inputs, train_dense_inputs, test_dense_inputs, \
    train_labels, test_labels = train_test_split(sparse_inputs, dense_inputs, labels, test_size=0.2, random_state=32)

    train_dataset = CustomDataset(train_sparse_inputs,train_dense_inputs,train_labels)
    test_dataset = CustomDataset(test_sparse_inputs,test_dense_inputs,test_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model = AFM(num_sparse_features,num_dense_features,num_categories_list,embedding_dim,att_embedding_dim,hidden_units)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss= 0.0

        for batch_sparse_inputs, batch_dense_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_sparse_inputs,batch_dense_inputs).squeeze()
            loss = criterion(outputs, batch_labels.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

        # 评估模型
    evaluate_model(model, test_loader)

