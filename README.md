# Movie Recommendation System with Graph Neural Networks (GNN)

This repository contains a PyTorch implementation of a movie recommendation system using Graph Neural Networks (GNN). The system models user-movie interactions as a bipartite graph, where users and movies are nodes, and edges represent user ratings for movies. By leveraging GNN layers, the model learns to predict user ratings for movies.

---

## Requirements
To run this project, install the following dependencies:
- Python 3.8+
- PySpark
- PyTorch
- PyTorch Geometric

## Libraries Used

### **PySpark** (Big Data Processing)
- **`pyspark.sql.SparkSession`**: Used to create a Spark session.
- **`pyspark.sql.functions`**: Includes utility functions for data manipulation, such as:
  - **`col`**: Specify a column in a DataFrame.
  - **`lit`**: Add constant values to a column.
  - **`isnan`**: Check for NaN values.
  - **`when`**: Create conditional expressions.
  - **`count`**: Count the number of elements in a column.

### **PyTorch** (Deep Learning)
- **`torch`**: Core PyTorch library for tensor operations.
- **`torch_geometric.data.Data`**: Stores graph data structures.
- **`torch_geometric.nn.GCNConv`**: Implements Graph Convolutional Network (GCN) layers.

### **PyTorch Neural Network Modules**
- **`torch.nn.Linear`**: Fully connected layer.
- **`torch.nn.ReLU`**: ReLU activation function.
- **`torch.nn.Dropout`**: Applies dropout regularization to prevent overfitting.
- **`torch.nn.BatchNorm1d`**: Batch normalization for stable and faster training.

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
