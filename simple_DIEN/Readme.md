This project implements a Deep Interest Evolution Network (DIEN) using PyTorch, which is designed to capture user interest evolution in recommendation systems. The model utilizes **Attention**, **AUGRU** (Gated Recurrent Unit with Attention), and a hierarchical structure to effectively model dynamic user behaviors over time.

## Project Structure

The project is divided into three main Python scripts:

1. **utils.py** - Handles data generation, preprocessing, and loading.
2. **DIEN_class.py** - Contains the implementation of the DIEN model, including attention mechanisms and the Interest Layer with AUGRU.
3. **DIEN_run.py** - Provides the training and evaluation process for the model.
