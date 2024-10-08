This project implements the **Attentional Factorization Machine (AFM)** model using **PyTorch**. AFM extends the traditional **Factorization Machine (FM)** by incorporating an **attention mechanism**, allowing the model to assign different levels of importance to different feature interactions. AFM is commonly used in recommendation systems and tasks where feature interactions are critical, such as click-through rate (CTR) prediction, user-item interaction, and more.

The **AFM** model includes:

1. **Embedding Layer**: Converts sparse (categorical) features into dense vector representations.
2. **Pairwise Interaction Layer**: Calculates the interactions between every pair of embedded features.
3. **Attention Mechanism**: Assigns attention weights to different feature interactions, learning which interactions are more important for the task at hand.
4. **Prediction Layer**: Aggregates the weighted interactions and combines them with a linear term to make the final prediction.
