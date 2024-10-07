## NFM

#### **Project Description**

This project implements the **Neural Factorization Machine (NFM)** using **PyTorch**. NFM is a powerful model designed for tasks that involve both sparse (categorical) and dense (numerical) features. It combines the benefits of traditional factorization machines with the representation learning ability of neural networks, making it particularly suitable for recommendation systems and tasks where feature interactions are important.

The **NFM model** consists of:

1. **Embedding Layer**: For sparse (categorical) features, converting them into dense vector representations.
2. **Bi-Interaction Layer**: Captures second-order interactions between the embedded features by combining them non-linearly, but avoids the computational cost of explicit pairwise interactions.
3. **Deep Neural Network (DNN)**: A stack of fully connected layers to further process the interactions and dense features.
4. **Linear Layer**: Directly processes the original features, helping the model memorize simple patterns.
