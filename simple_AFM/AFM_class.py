import torch
import torch.nn as nn

class AFM(nn.Module):
    def __init__(self, num_sparse_features, num_dense_features, num_categories_list, embedding_dim,\
                 att_embedding_dim,hidden_units):
        super(AFM, self).__init__()

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_categories_list[i],embedding_dim) for i in range(num_sparse_features)]
        )
        # 注意力权重向量
        self.attention_W = nn.Parameter(torch.randn(embedding_dim,att_embedding_dim))
        self.attention_b = nn.Parameter(torch.randn(att_embedding_dim))
        self.attention_h = nn.Parameter(torch.randn(att_embedding_dim))

        #线性层
        self.linear = nn.Linear(num_sparse_features + num_dense_features, 1)
        #一层dnn
        self.dnn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,1)
        )

    def forward(self, sparse_inputs, dense_inputs):
        sparse_embeds = [self.embeddings[i](sparse_inputs[:,i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = torch.stack(sparse_embeds, dim = 1) # (batch_size, num_features, embedding_dim)

        num_features = sparse_embeds.shape[1]
        pairwise_interactions = []
        for i in range(num_features):
            for j in range(i+1,num_features):
                pairwise_interactions.append(sparse_embeds[:,i] * sparse_embeds[:,j])

        pairwise_interactions = torch.stack(pairwise_interactions,dim = 1) # (batch_size, n(n-1)/2, embedding_dim)

        # 通过注意力机制
        attention_scores = torch.tanh(torch.matmul(pairwise_interactions,
                                                   self.attention_W) + self.attention_b)  # (batch_size, n(n-1)/2, att_embedding_dim)

        attention_weights = torch.softmax(torch.matmul(attention_scores,self.attention_h),dim = 1) # (batch_size, n(n-1)/2)
        # 将注意力权重应用到 pairwise_interactions
        weighted_interactions = (pairwise_interactions * attention_weights.unsqueeze(-1)).sum(dim = 1) #(batch_size, embedding_dim)

        # DNN 部分处理交互结果
        dnn_output = self.dnn(weighted_interactions)

        linear_output = self.linear(torch.cat([sparse_inputs.float(),dense_inputs], dim = 1))

        output = linear_output+dnn_output


        return torch.sigmoid(output)