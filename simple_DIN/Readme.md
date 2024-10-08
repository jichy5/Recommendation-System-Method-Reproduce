Deep Interest Network (DIN) is a deep learning model designed for personalized recommendation tasks. It uses user interaction history (e.g., products or content the user has previously interacted with) to model the user's interest dynamically. The core idea of DIN is to apply attention mechanisms to capture the relevance between the user's history and the candidate item, then make personalized recommendations.

This implementation includes the following:

- **DataLoader**: Handles the data preprocessing and loading.
- **DIN Model**: The main model, including embedding layers, attention mechanisms, and a fully connected neural network (DNN).
- **AttentionPoolingLayer**: Implements the attention mechanism to weight user history based on the relevance to the current candidate item.
