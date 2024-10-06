 DSIN（Deep Session Interest Network）模型，使用 PyTorch 框架，包含以下三个文件：

- `utils.py`：包含数据预处理和辅助函数。
- `DISN_class.py`：定义了 DSIN 模型及其相关组件。
- `DISN_run.py`：主运行脚本，用于训练和测试模型。
- 其中DISN_class里用到了transformer 提取兴趣， bilstm提取兴趣的变迁和 activation_layers 探究兴趣和电影id的相似度。

This project implements the Deep Session Interest Network (DSIN) model using PyTorch. It includes the following three files:

- `utils.py`: Contains data preprocessing and utility functions.
- `DISN_class.py`: Defines the DSIN model and its related components.
- `DISN_run.py`: The main script for training and testing the model.
