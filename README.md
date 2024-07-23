# Recommendation System

## Overview

This project implements a recommendation system using various algorithms. The goal is to evaluate the performance of different recommendation techniques and compare them using the MovieLens dataset. The system includes algorithms such as SVD, SVD++, ContentKNN, RBM, AutoRec, and a hybrid model that combines multiple approaches.

## Installation

Install my-project with npm

```python
  pip install numpy scipy scikit-learn tensorflow surprise
```
## Usage

To run the recommendation system, first download files and do installation and then execute the main script:

```python
python recommendation_system.py
```
The script will:

- Load the MovieLens dataset.
- Initialize and train various recommendation algorithms.
- Evaluate each algorithm's performance.
- Print evaluation results.

## Data

### MovieLens Dataset
The dataset includes two main files:

**Movies file (movies.csv)**: Contains movie information.

| movieId | title                        | genres                              |
|---------|------------------------------|-------------------------------------|
| 1       | Toy Story (1995)             | Adventure, Animation, Children, Comedy, Fantasy |
| 2       | Jumanji (1995)               | Adventure, Children, Fantasy          |
| 3       | Grumpier Old Men (1995)      | Comedy, Romance                       |
| 4       | Waiting to Exhale (1995)     | Comedy, Drama, Romance                |
| 5       | Father of the Bride Part II (1995) | Comedy                          |
| 6       | Heat (1995)                  | Action, Crime, Thriller              |
| 7       | Sabrina (1995)               | Comedy, Romance                      |

**Ratings file (ratings.csv)**: Contains user ratings.

| userId | movieId | rating | timestamp   |
|--------|---------|--------|-------------|
| 1      | 31      | 2.5    | 1260759144  |
| 1      | 1029    | 3.0    | 1260759179  |
| 1      | 1061    | 3.0    | 1260759182  |
| 1      | 1129    | 2.0    | 1260759185  |

## Files
Here's a brief overview of the files in the repository:

- **AutoRec.py**: Implementation of the AutoRec algorithm.
- **AutoRecAlgorithm.py**: Supporting functions for the AutoRec algorithm.
- **ContentKNNAlgorithm.py**: Implementation of the ContentKNN algorithm.
- **EvaluatedAlgorithm.py**: Contains evaluation logic for algorithms.
- **EvaluationData.py**: Data handling for evaluation purposes.
- **Evaluator.py**: Class for evaluating the performance of recommendation algorithms.
- **HybridAlgorithm.py**: Implementation of the hybrid recommendation algorithm.
- **MovieLens.py**: Data loading and preprocessing for the MovieLens dataset.
- **RBM.py**: Core logic for Restricted Boltzmann Machine.
- **RBMAlgorithm.py**: Implementation of the RBM algorithm.
- **README.md**: This file.
- **RecSys.py**: Main script for running the recommendation system.
- **RecommenderMetrics.py**: Metrics and evaluation functions for recommendation algorithms.
- **movies.csv**: Movie data for the MovieLens dataset.
- **ratings.csv**: User ratings data for the MovieLens dataset.
## Algorithm
### 1. SVD (Singular Value Decomposition)

**Description**: Factorizes the user-item matrix into singular value matrices.

**Implementation**: SVD from the surprise library.

### 2. SVD++ (SVD Plus Plus)

**Description**: An extension of SVD that incorporates implicit feedback.

**Implementation**: SVDpp from the surprise library.

### 3. ContentKNN

**Description**: Uses content-based features and K-nearest neighbors for recommendations.

**Implementation**: ContentKNNAlgorithm class.

### 4. RBM (Restricted Boltzmann Machine)

**Description**: A type of neural network that can be used for collaborative filtering.

**Implementation**: RBMAlgorithm class.

### 5. AutoRec

**Description**: An autoencoder-based recommendation system.

**Implementation**: AutoRecAlgorithm class.

### 6. Hybrid

**Description**: Combines multiple recommendation algorithms to improve performance.

**Implementation**: HybridAlgorithm class.

### 7. Random

**Description**: Provides random recommendations as a baseline.

**Implementation**: NormalPredictor from the surprise library.
## Algorithm Comparision(Output)

![Screenshot](https://github.com/ParitKansal/Recommendation-System/blob/main/Result%20of%20recommendation%20app.png)

## Authors

- [@ParitKansal](https://github.com/ParitKansal)

