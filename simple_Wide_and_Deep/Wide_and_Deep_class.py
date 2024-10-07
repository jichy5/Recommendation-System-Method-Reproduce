# model.py
import torch
import torch.nn as nn
import itertools


class WideAndDeepModel(nn.Module):
    def __init__(self, num_categories, embedding_dim, hidden_units, num_sparse_features, num_dense_features):
        super(WideAndDeepModel, self).__init__()
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.num_sparse_features = num_sparse_features
        self.num_dense_features = num_dense_features

        #构建交互的特征放在Wide模型中
        self.cross_features_indices = []
        for r in range(2, self.num_sparse_features + 1):
            combinations = list(itertools.combinations(range(self.num_sparse_features),r))
            self.cross_features_indices.extend(combinations)


        # 计算交叉特征的数量，这里我为了简便把所有sparse_feature的特征值都设置为num_categories个可能值
        if len(self.cross_features_indices) > 0:
            cross_feature_length = len(self.cross_features_indices[0])
            self.num_cross_features = 0
            for idx_tuple in self.cross_features_indices:
                self.num_cross_features += self.num_categories ** len(idx_tuple)
        else:
            self.num_cross_features = 0


        #  Wide 部分的线性层
        wide_input_dim = self.num_categories * self.num_sparse_features + self.num_cross_features + self.num_dense_features
        self.wide_linear = nn.Linear(wide_input_dim, 1)

        #  Deep 部分的嵌入层和 DNN
        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.num_categories, self.embedding_dim) for _ in range(self.num_sparse_features)]
        )
        deep_input_dim = self.embedding_dim * self.num_sparse_features + self.num_dense_features
        self.deep_dnn = nn.Sequential(
            nn.Linear(deep_input_dim, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, 1)
        )

    def forward(self, sparse_inputs, dense_inputs):
        #wide one-hot编码
        one_hot_encoded = []
        for i in range(self.num_sparse_features):
            one_hot_encoded.append(torch.nn.functional.one_hot(sparse_inputs[:,i], num_classes = self.num_categories))
        one_hot_encoded = torch.cat(one_hot_encoded, dim= 1).float()

        # 交叉特征的 one-hot 编码
        cross_one_hot_encoded = []
        for idx_tuple in self.cross_features_indices:
            # 计算组合特征的索引
            cross_feature = sparse_inputs[:, idx_tuple[0]].clone()
            #这里类似于这种做法类似于将多个特征值“拼接”成一个新的组合索引。
            # 例如：若 num_categories = 4，稀疏特征 [0, 1] 组合后变为 0*4 + 1 = 1。
            #对于三阶组合 (0,1,2)，若 sparse_inputs[:,0]=0, sparse_inputs[:,1]=1,
            # sparse_inputs[:,2]=2，组合后的索引为 0*4*4 + 1*4 + 2 = 6。
            for idx in idx_tuple[1:]:
                cross_feature = cross_feature * self.num_sparse_features + sparse_inputs[:, idx]
            num_classes = self.num_categories ** len(idx_tuple)
            cross_one_hot = torch.nn.functional.one_hot(cross_feature,num_classes = num_classes).float()
            cross_one_hot_encoded.append(cross_one_hot)
        if cross_one_hot_encoded:
            cross_one_hot_encoded = torch.cat(cross_one_hot_encoded, dim= 1)
            wide_input = torch.cat([one_hot_encoded, cross_one_hot_encoded,dense_inputs], dim = 1)
        else:
            wide_input = torch.cat([one_hot_encoded, dense_inputs], dim= 1)

        wide_output = self.wide_linear(wide_input)

        # Deep 部分
        embedded_sparse = [self.embeddings[i](sparse_inputs[:, i]) for i in range(self.num_sparse_features)]
        embedded_sparse = torch.cat(embedded_sparse, dim=1)
        deep_input = torch.cat([embedded_sparse,dense_inputs],  dim = 1)
        deep_output = self.deep_dnn(deep_input)


        output = wide_output + deep_output

        return torch.sigmoid(output)
