import torch
import torch.nn as nn

class BiInteractionPooling(nn.Module):
    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, embedded_features):
        # embedded_features 的形状为 (batch_size, num_features, embedding_dim)
        sum_square = torch.pow(torch.sum(embedded_features, dim = 1), 2) # (batch_size, embedding_dim)
        square_sum = torch.sum(torch.pow(embedded_features, 2), dim = 1) # (batch_size, embedding_dim)

        # 计算 f_BI(V_x)
        bi_interaction = 0.5 * (sum_square - square_sum)

        return bi_interaction


class NFM(nn.Module):
    def __init__(self, num_sparse_features, num_dense_features, num_categories_list, embedding_dim, hidden_units_list, dropout_rate = 0.5,num_hidden_layers=3):
        super(NFM, self).__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_categories_list[i], embedding_dim) for i in range(num_sparse_features)]
        )

        self.bi_interaction_pooling = BiInteractionPooling()

        self.bi_interaction_bn = nn.BatchNorm1d(embedding_dim) #对于每个维度 i（embedding_dim 中的某个维度）进行归一化
        self.dropout = nn.Dropout(dropout_rate)

        # 线性层
        self.linear = nn.Linear(num_sparse_features + num_dense_features, 1)

        dnn_layers = []
        input_dim = embedding_dim + num_dense_features

        for hidden_units in hidden_units_list:
            dnn_layers.append(nn.Linear(input_dim,hidden_units))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_units# 每层输入维度是上一层的输出维度

        dnn_layers.append(nn.Linear(hidden_units_list[-1], 1))
        self.dnn = nn.Sequential(*dnn_layers)




    def forward(self, spare_inputs, dense_inputs):
        spare_inputs = [self.embedding[i](spare_inputs[:,i]) for i in range(spare_inputs.shape[1])]
        spare_inputs = torch.stack(sparse_embeds, dim = 1)
    def forward(self, sparse_inputs, dense_inputs):
        # 稀疏特征通过 embedding 层
        sparse_embeds = [self.embeddings[i](sparse_inputs[:,i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = torch.stack(sparse_embeds, dim = 1)

        # 双线性池化层
        interaction_output = self.bi_interaction_pooling(sparse_embeds)
        interaction_output = self.bi_interaction_bn(interaction_output)
        interaction_output = self.dropout(interaction_output)

        linear_part = self.linear(torch.cat([sparse_inputs.float(), dense_inputs], dim = 1))

        # DNN 部分，将 dense 特征与 interaction_output 拼接
        dnn_input = torch.cat([interaction_output,dense_inputs], dim = 1)
        dnn_output = self.dnn(dnn_input)
        # 将线性部分与 DNN 部分的输出相加
        output = linear_part + dnn_output

        return torch.sigmoid(output)
