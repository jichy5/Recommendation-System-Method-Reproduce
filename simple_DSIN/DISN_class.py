import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import VarLenSparseFeat, DenseFeat,SparseFeat


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        sigmoid_x = torch.sigmoid(x)
        return self.alpha * (1 - sigmoid_x) * x + sigmoid_x * x

class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_units, embedding_dim, activation='prelu'):
        super(LocalActivationUnit, self).__init__()
        layers = []
        input_dim = 4 * embedding_dim
        for unit in hidden_units:
            layers.append(nn.Linear(input_dim, unit))
            if activation == 'prelu':
                layers.append(nn.PReLU())
            elif activation == 'dice':
                layers.append(Dice())
            else:
                layers.append(nn.ReLU())
            input_dim = unit
        self.dnn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, keys):
        # query: [batch_size, 1, embedding_dim]
        # keys: [batch_size, seq_len, embedding_dim]
        seq_len = keys.size(1)
        queries = query.expand(-1, seq_len, -1)
        att_input = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
        att_out = self.dnn(att_input)
        att_out = self.fc(att_out)
        att_out = att_out.squeeze(-1)  # [batch_size, seq_len]
        return att_out

class AttentionPoolingLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_units, activation='prelu', return_score=False):
        super(AttentionPoolingLayer, self).__init__()
        self.local_att = LocalActivationUnit(hidden_units, embedding_dim, activation)
        self.return_score = return_score

    def forward(self, query, keys, keys_length):
        # query: [batch_size, 1, embedding_dim]
        # keys: [batch_size, seq_len, embedding_dim]
        attention_score = self.local_att(query, keys)  # [batch_size, seq_len]
        # Mask
        mask = torch.arange(keys.size(1)).unsqueeze(0).to(keys.device) < keys_length.unsqueeze(1)
        attention_score = attention_score.masked_fill(~mask, float('-inf'))
        attention_score = F.softmax(attention_score, dim=1)
        if not self.return_score:
            output = torch.bmm(attention_score.unsqueeze(1), keys)  # [batch_size, 1, embedding_dim]
            output = output.squeeze(1)  # [batch_size, embedding_dim]
            return output
        else:
            return attention_score

class Transformer(nn.Module):
    def __init__(self, embedding_size, singlehead_emb_size=8, att_head_nums=1, dropout_rate=0.0, use_positional_encoding=False,
                 use_res=True, use_feed_forward=True, use_layer_norm=False, blinding=False):
        super(Transformer, self).__init__()
        self.singlehead_emb_size = singlehead_emb_size
        self.att_head_nums = att_head_nums
        self.num_units = self.singlehead_emb_size * self.att_head_nums
        self.use_res = use_res
        self.use_feed_forward = use_feed_forward
        self.dropout_rate = dropout_rate
        self.use_positional_encoding = use_positional_encoding
        self.use_layer_norm = use_layer_norm
        self.blinding = blinding  # 如果为True，则在计算注意力时屏蔽未来信息
        self.dropout = nn.Dropout(dropout_rate)

        self.W_Query = nn.Linear(embedding_size, self.num_units)
        self.W_Key = nn.Linear(embedding_size, self.num_units)
        self.W_Value = nn.Linear(embedding_size, self.num_units)

        if use_feed_forward:
            self.fw1 = nn.Linear(self.num_units, 4 * self.num_units)
            self.fw2 = nn.Linear(4 * self.num_units, self.num_units)
        if use_layer_norm:
            self.ln = nn.LayerNorm(self.num_units)

    def forward(self, queries, keys, keys_mask=None):
        # queries: [batch_size, seq_len_q, embedding_size]
        # keys: [batch_size, seq_len_k, embedding_size]
        # keys_mask: [batch_size, seq_len_k]
        Q = self.W_Query(queries)  # [batch_size, seq_len_q, num_units]
        K = self.W_Key(keys)       # [batch_size, seq_len_k, num_units]
        V = self.W_Value(keys)     # [batch_size, seq_len_k, num_units]

        # Split heads
        Q_ = Q.view(Q.size(0), Q.size(1), self.att_head_nums, self.singlehead_emb_size).permute(0, 2, 1, 3)
        K_ = K.view(K.size(0), K.size(1), self.att_head_nums, self.singlehead_emb_size).permute(0, 2, 1, 3)
        V_ = V.view(V.size(0), V.size(1), self.att_head_nums, self.singlehead_emb_size).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q_, K_.transpose(-1, -2)) / (self.singlehead_emb_size ** 0.5)

        if keys_mask is not None:
            keys_mask_expanded = keys_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len_k]
            attn_scores = attn_scores.masked_fill(~keys_mask_expanded, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        outputs = torch.matmul(attn_probs, V_)  # [batch_size, att_head_nums, seq_len_q, singlehead_emb_size]

        # Concatenate heads
        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        outputs = outputs.view(outputs.size(0), outputs.size(1), -1)  # [batch_size, seq_len_q, num_units]

        if self.use_res:
            outputs += queries
        if self.use_layer_norm:
            outputs = self.ln(outputs)
        if self.use_feed_forward:
            outputs_ff = self.fw1(outputs)
            outputs_ff = F.relu(outputs_ff)
            outputs_ff = self.dropout(outputs_ff)
            outputs_ff = self.fw2(outputs_ff)
            if self.use_res:
                outputs_ff += outputs
            if self.use_layer_norm:
                outputs_ff = self.ln(outputs_ff)
            outputs = outputs_ff
        return outputs

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=None, layers=1, res_layers=0, dropout_rate=0.0, merge_mode='ave'):
        super(BiLSTM, self).__init__()
        if hidden_size is None:
            hidden_size = input_size
        self.layers = layers
        self.res_layers = res_layers
        self.merge_mode = merge_mode
        self.lstm_fw = nn.ModuleList()
        self.lstm_bw = nn.ModuleList()
        for _ in range(layers):
            self.lstm_fw.append(nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False))
            self.lstm_bw.append(nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        input_fw = x
        input_bw = x
        for i in range(self.layers):
            output_fw, _ = self.lstm_fw[i](input_fw)
            output_bw, _ = self.lstm_bw[i](torch.flip(input_bw, dims=[1]))
            output_bw = torch.flip(output_bw, dims=[1])
            if i >= self.layers - self.res_layers:
                output_fw = output_fw + input_fw
                output_bw = output_bw + input_bw
            input_fw = self.dropout(output_fw)
            input_bw = self.dropout(output_bw)
        if self.merge_mode == 'fw':
            output = output_fw
        elif self.merge_mode == 'bw':
            output = output_bw
        elif self.merge_mode == 'concat':
            output = torch.cat([output_fw, output_bw], dim=-1)
        elif self.merge_mode == 'sum':
            output = output_fw + output_bw
        elif self.merge_mode == 'ave':
            output = (output_fw + output_bw) / 2
        elif self.merge_mode == 'mul':
            output = output_fw * output_bw
        else:
            output = [output_fw, output_bw]
        return output

class DSIN(nn.Module):
    def __init__(self, feature_columns, sess_feature_list, bias_encoding=True, singlehead_emb_size=8,
                 att_head_nums=1, dnn_hidden_units=(64, 32)):
        super(DSIN, self).__init__()
        self.feature_columns = feature_columns
        self.sess_feature_list = sess_feature_list
        self.bias_encoding = bias_encoding

        # 构建embedding层
        self.embedding_layer_dict = nn.ModuleDict()
        for fc in self.feature_columns:
            if isinstance(fc, tuple) and fc[0] == 'sess_length':
                continue
            if hasattr(fc, 'vocabulary_size'):  #因为只有稀疏特征和变长稀疏特征才具有 vocabulary_size
                if isinstance(fc, VarLenSparseFeat):
                    self.embedding_layer_dict[fc.name] = nn.Embedding(fc.vocabulary_size + 1, fc.embedding_dim, padding_idx=0)
                else:
                    self.embedding_layer_dict[fc.name] = nn.Embedding(fc.vocabulary_size, fc.embedding_dim)

        # 自动获取特征名称列表
        self.dense_features = [fc.name for fc in self.feature_columns if isinstance(fc, DenseFeat)]
        self.sparse_features = [fc.name for fc in self.feature_columns if isinstance(fc, SparseFeat)]
        self.varlen_sparse_features = [fc.name for fc in self.feature_columns if isinstance(fc, VarLenSparseFeat)]

        # 计算hist_emb_size  #我们的例子只有一个VarLen
        self.hist_emb_size = sum(
            fc.embedding_dim for fc in self.feature_columns
            if isinstance(fc, VarLenSparseFeat)
        )

        if singlehead_emb_size * att_head_nums != self.hist_emb_size:
            raise ValueError('hist_emb_size must equal to singlehead_emb_size * att_head_nums')

        # 定义Transformer
        self.self_attention = Transformer(
            self.hist_emb_size,
            singlehead_emb_size=singlehead_emb_size,
            att_head_nums=att_head_nums,
            dropout_rate=0,
            use_layer_norm=True,
            use_positional_encoding=not bias_encoding,
            blinding=False
        )

        # BiLSTM
        self.bilstm = BiLSTM(self.hist_emb_size, layers=1, res_layers=0, dropout_rate=0.0)

        # AttentionPoolingLayer
        self.attention_pooling1 = AttentionPoolingLayer(self.hist_emb_size, hidden_units=(32, 16), activation='prelu')
        self.attention_pooling2 = AttentionPoolingLayer(self.hist_emb_size, hidden_units=(32, 16), activation='prelu')

        # DNN
        input_dim = len(self.dense_features)
        input_dim += sum([fc.embedding_dim for fc in self.feature_columns if isinstance(fc, SparseFeat)])
        input_dim += self.hist_emb_size * 2
        dnn_layers = []
        for dim in dnn_hidden_units:
            dnn_layers.append(nn.Linear(input_dim, dim))
            dnn_layers.append(nn.PReLU())
            dnn_layers.append(nn.Dropout(0.2))
            input_dim = dim
        dnn_layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, inputs):
        # 处理密集特征
        dense_inputs = [inputs[feat].float().unsqueeze(1) for feat in self.dense_features]
        if dense_inputs:
            dense_inputs = torch.cat(dense_inputs, dim=1)
        else:
            dense_inputs = torch.tensor([], device=next(self.parameters()).device)

        # 处理稀疏特征
        sparse_embeddings = []
        for feat in self.sparse_features:
            emb = self.embedding_layer_dict[feat](inputs[feat])
            emb = emb.view(emb.size(0), -1)
            sparse_embeddings.append(emb)
        if sparse_embeddings:
            sparse_inputs = torch.cat(sparse_embeddings, dim=1)
            #[batch_size, embedding_dim_1 + embedding_dim_2 + ... + embedding_dim_n]
        else:
            sparse_inputs = torch.tensor([], device=next(self.parameters()).device)

        # 处理query embedding
        query_embeddings = []
        for feat in self.sess_feature_list:
            emb = self.embedding_layer_dict[feat](inputs[feat])
            query_embeddings.append(emb)
        query_emb = torch.cat(query_embeddings, dim=-1)
        query_emb = query_emb.unsqueeze(1)  # [batch_size, 1, emb_dim]

        # 处理会话
        sess_emb = self.embedding_layer_dict['sess_array'](inputs['sess_array'])  # [batch_size, sess_max_count, seq_len, emb_dim]

        batch_size, sess_count, seq_len, emb_dim = sess_emb.size()
        sess_emb = sess_emb.view(batch_size * sess_count, seq_len, emb_dim)
        # 获取序列长度mask
        seq_lengths = inputs['sess_lengths'].view(-1)  # [batch_size * sess_max_count]
        # 处理序列长度为零的情况
        seq_lengths_clamped = seq_lengths.clone()
        seq_lengths_clamped[seq_lengths_clamped == 0] = 1 #让他们至少还有一个维度
        keys_mask = torch.arange(seq_len).to(sess_emb.device).unsqueeze(0) < seq_lengths_clamped.unsqueeze(1)

        # 通过Transformer
        tr_out = self.self_attention(sess_emb, sess_emb, keys_mask)
        tr_out = tr_out.mean(dim=1)  # [batch_size * sess_max_count, emb_dim]
        #将整个序列的特征聚合成一个单一的向量，作为整个会话的特征表示
        sess_fea = tr_out.view(batch_size, sess_count, -1)  # [batch_size, sess_count, emb_dim]

        # BiLSTM
        lstm_output = self.bilstm(sess_fea)  # [batch_size, sess_count, emb_dim]

        # AttentionPoolingLayer
        user_sess_len = inputs['sess_length']
        # 创建会话mask
        sess_mask = torch.arange(sess_count)[None, :].to(sess_fea.device) < user_sess_len[:, None]
        # interest_attention和lstm_attention
        interest_attention = self.attention_pooling1(query_emb, sess_fea, user_sess_len)
        lstm_attention = self.attention_pooling2(query_emb, lstm_output, user_sess_len)

        # 拼接
        deep_input = torch.cat([dense_inputs, sparse_inputs, interest_attention, lstm_attention], dim=-1)

        # DNN
        output = self.dnn(deep_input)
        output = torch.sigmoid(output)
        return output.squeeze(-1)
