# Movie Recommendation System with Graph Neural Networks (GNN)

This repository contains a PyTorch implementation of a movie recommendation system using Graph Neural Networks (GNN). The system models user-movie interactions as a bipartite graph, where users and movies are nodes, and edges represent user ratings for movies. By leveraging GNN layers, the model learns to predict user ratings for movies.

---

## Features
- **Graph Neural Network (GNN)**: Utilizes GCNConv layers for graph-based learning.
- **Edge Attributes**: Encodes user ratings as edge weights.
- **Batch Normalization**: Improves convergence and stability.
- **Dropout Regularization**: Reduces overfitting.
- **Huber Loss**: Handles outlier ratings effectively.
- **Early Stopping**: Avoids overfitting by monitoring validation metrics.
- **Learning Rate Scheduler**: Adjusts learning rate based on performance.

---

## Requirements
To run this project, install the following dependencies:
- Python 3.8+
- PyTorch
- PyTorch Geometric

## Dataset
This implementation expects a dataset in the form of a DataFrame with the following columns:
- `userId`: Unique ID of the user.
- `movieId`: Unique ID of the movie.
- `rating`: Rating given by the user to the movie.

Example structure:
| userId | movieId | rating |
|--------|---------|--------|
| 1      | 101     | 4.5    |
| 2      | 102     | 3.0    |

---

## How It Works
1. **Data Preprocessing**:
   - Convert `userId` and `movieId` to unique indices.
   - Create an edge list representing user-movie interactions.
   - Normalize ratings to use as edge attributes.

2. **Model Architecture**:
   - Three GCNConv layers with batch normalization and ReLU activation.
   - Fully connected layers for regression to predict user ratings.

3. **Training**:
   - The model is trained to minimize the Huber Loss.
   - RMSE and MAE are used as evaluation metrics.
   - Early stopping is implemented to prevent overfitting.


## Results
- **Metrics**: The model evaluates performance using RMSE and MAE.
- **Early Stopping**: Stops training if no improvement in RMSE is observed for 20 consecutive epochs.

---

## Acknowledgments
This implementation is inspired by PyTorch Geometric and utilizes graph-based learning techniques for recommendation systems. Special thanks to the open-source community for their valuable resources.
