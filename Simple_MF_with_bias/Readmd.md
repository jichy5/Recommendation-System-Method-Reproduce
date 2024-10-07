## MF

该项目实现了一个 **矩阵分解 (MF)** 模型，这是推荐系统中常用的一种协同过滤技术。项目包括两个文件：

1. `mf_model.py`：定义了MF模型类，包含用户和物品的嵌入、偏置项，以及用于预测用户-物品评分的 `forward` 方法。
2. `train_test.py`：生成用户-物品评分矩阵，将其拆分为训练集和测试集，然后对训练集进行模型训练，同时使用 **均方根误差 (RMSE)** 和 **平均绝对误差 (MAE)** 等指标对测试集进行评估。

### Overview

This repository contains the implementation of a **Matrix Factorization (MF)** model, a collaborative filtering technique used for recommendation systems. The project includes two files:

1. `mf_model.py`: Defines the MF model class with user and item embeddings, biases, and a forward method to predict user-item ratings.
2. `train_test.py`: Generates a user-item rating matrix, splits it into training and testing sets, and then trains the MF model on the training set. It also evaluates the model performance on the testing set using metrics like **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.
