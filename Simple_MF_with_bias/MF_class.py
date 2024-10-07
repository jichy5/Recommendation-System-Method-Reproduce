import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size = 20):
        super(MF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

        # 用户和物品的偏置项
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # 全局偏置项
        self.global_bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, user_id, item_id):
        user_embedding = self.user_embedding(user_id) # shape: (batch_size, embedding_dim)
        item_embedding = self.item_embedding(item_id)

        user_bias = self.user_bias(user_id).squeeze() # shape: (batch_size)
        item_bias = self.item_bias(item_id).squeeze()

        # 计算用户和物品 embedding 的内积
        dot_product = torch.mul(user_embedding,item_embedding).sum(dim=1)

        prediction = self.global_bias + user_bias + item_bias + dot_product
        return prediction



