import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import namedtuple

# 定义特征结构
SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'maxlen', 'length_name'])


def generate_hist_sess(max_sess_count, max_seq_len):
    """生成历史会话数据"""
    sess_list = []
    sess_lengths = []
    sess_length = np.random.randint(1, max_sess_count + 1)
    for _ in range(sess_length):
        seq_len = np.random.randint(1, max_seq_len + 1)
        sess = np.random.randint(1, 20, size=seq_len).tolist()
        sess += [0] * (max_seq_len - seq_len)  # 填充到相同长度
        sess_list.append(sess)
        sess_lengths.append(seq_len)
    # 如果会话数量不足max_sess_count，填充空会话
    for _ in range(max_sess_count - sess_length):
        sess_list.append([0] * max_seq_len)
        sess_lengths.append(0)
    return sess_list, sess_lengths, sess_length


def preprocess_data(samples_data, sparse_features, dense_features, varlen_sparse_features, max_seq_len):
    """对数据进行预处理，包括特征映射和构建特征列"""
    # 为稀疏特征创建索引映射
    for feat in sparse_features:
        lbe = LabelEncoder()
        samples_data[feat] = lbe.fit_transform(samples_data[feat])

    # 对密集特征进行归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    samples_data[dense_features] = mms.fit_transform(samples_data[dense_features])

    # 对于变长稀疏特征，构建索引映射
    sess_values = []
    for sess_array in samples_data['sess_array']:
        for seq in sess_array:
            sess_values.extend(seq)
    sess_values = [val for val in sess_values if val != 0]  # 去除填充的0

    unique_values = np.unique(sess_values)
    value2index = {value: idx + 1 for idx, value in enumerate(unique_values)}  # 从1开始，0留给padding
    value2index[0] = 0  # 保留0

    # 映射sess_array中的值
    def map_seq(seq):
        return [value2index.get(item, 0) for item in seq]

    def map_sess_array(sess_array):
        return [map_seq(seq) for seq in sess_array]

    #所有的电影 ID 都被映射为索引
    samples_data['sess_array'] = samples_data['sess_array'].apply(map_sess_array)

    # 特征封装
    feature_columns = []

    # 稀疏特征
    for feat in sparse_features:
        vocab_size = samples_data[feat].nunique()
        feature_columns.append(SparseFeat(feat, vocab_size, embedding_dim=8))

    # 密集特征
    for feat in dense_features:
        feature_columns.append(DenseFeat(feat, 1))

    # 变长稀疏特征
    for feat in varlen_sparse_features:
        vocab_size = len(value2index)
        feature_columns.append(
            VarLenSparseFeat(feat, vocab_size, embedding_dim=8, maxlen=max_seq_len, length_name='sess_lengths'))

    # 添加 'sess_length' 特征
    feature_columns.append('sess_length')

    return samples_data, feature_columns


def prepare_input(data, sparse_features, dense_features, varlen_sparse_features):
    """准备模型输入数据"""
    X = {}
    for feat in sparse_features + dense_features + ['sess_length']:
        X[feat] = data[feat].values
    for feat in varlen_sparse_features:
        X[feat] = np.array(data[feat].tolist())  # [batch_size, max_sess_count, max_seq_len]
    X['sess_lengths'] = np.array(data['sess_lengths'].tolist())  # [batch_size, max_sess_count]
    y = data['label'].values
    return X, y
