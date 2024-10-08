import torch
import torch.nn as nn

# 注意力池化层
class AttentionPoolingLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionPoolingLayer, self).__init__()
        self.attention_dense = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, query, keys):
        # query: (batch_size, embedding_dim)
        # keys: (batch_size, history_len, embedding_dim)
        # 扩展 query 的维度以匹配 keys
        query = query.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        attention_scores = torch.tanh(self.attention_dense(keys * query))  # (batch_size, history_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, history_len, 1)
        output = torch.sum(attention_weights * keys, dim=1)  # (batch_size, embedding_dim)
        return output

# DIN 模型
class DIN(nn.Module):
    def __init__(self, feature_sizes, embedding_dim, hidden_units, dropout_rate):
        super(DIN, self).__init__()

        self.embedding_dim = embedding_dim

        self.user_num_linear = nn.Linear(feature_sizes['user_num'], embedding_dim)
        self.user_cat_embedding = nn.ModuleDict({
            feature: nn.Embedding(feature_sizes['user_cat'][feature], embedding_dim)
            for feature in feature_sizes['user_cat']
        })

        self.item_num_linear = nn.Linear(feature_sizes['item_num'], embedding_dim)
        self.item_cat_embedding = nn.ModuleDict({
            feature: nn.Embedding(feature_sizes['item_cat'][feature], embedding_dim)
            for feature in feature_sizes['item_cat']
        })

        self.history_embedding = nn.Embedding(feature_sizes['history_item'], embedding_dim, padding_idx=0)


        self.attention = AttentionPoolingLayer(embedding_dim)

        self.dnn = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Linear(hidden_units // 2, 1)
        )

    def forward(self, user_num, user_cat, item_num, item_cat, history_items):
        #这里等于每个类型的特征我把它累加成一个embedding_dim的特征
        # 用户数值特征
        user_num_emb = self.user_num_linear(user_num)  # (batch_size, embedding_dim)

        # 用户类别特征
        user_cat_emb = sum([emb(user_cat[:, i]) for i, emb in enumerate(self.user_cat_embedding.values())])  # (batch_size, embedding_dim)

        # 用户嵌入
        user_emb = user_num_emb + user_cat_emb  # (batch_size, embedding_dim)

        # 物品数值特征
        item_num_emb = self.item_num_linear(item_num)  # (batch_size, embedding_dim)

        # 物品类别特征
        item_cat_emb = sum([emb(item_cat[:, i]) for i, emb in enumerate(self.item_cat_embedding.values())])  # (batch_size, embedding_dim)

        # 物品嵌入
        item_emb = item_num_emb + item_cat_emb  # (batch_size, embedding_dim)

        # 历史行为嵌入
        history_emb = self.history_embedding(history_items)  # (batch_size, history_len, embedding_dim)

        # 注意力池化
        history_attention_output = self.attention(item_emb, history_emb)  # (batch_size, embedding_dim)

        # 拼接特征
        combined_features = torch.cat([user_emb, item_emb, history_attention_output], dim=-1)  # (batch_size, embedding_dim * 3)

        # DNN 输出
        logits = self.dnn(combined_features)

        return torch.sigmoid(logits).squeeze(-1)
