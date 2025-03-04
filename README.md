# Movie-Recommendation-System-
A movie recommendation system using collaborative filtering on the MovieLens 100K dataset. Implements item-item similarity with Cosine Similarity and Pearson Correlation. Includes data preprocessing, hyperparameter tuning, and evaluation metrics (MAE, Precision, Recall) to optimize personalized recommendations.
# Movie Recommendation System

## Overview
This project implements a **movie recommendation system** using collaborative filtering techniques. The system predicts user preferences based on the **MovieLens 100K dataset**, applying **item-item similarity** using **Cosine Similarity** and **Pearson Correlation**.

## Features
- **Collaborative Filtering:** Item-based similarity approach
- **Similarity Measures:** Cosine Similarity & Pearson Correlation
- **Data Preprocessing:** Handles missing values and filters dataset based on user-defined parameters
- **Evaluation Metrics:** Mean Absolute Error (MAE), Precision, Recall
- **Experiments:** Hyperparameter tuning for **N (neighbors), T (train-test split), and M, M' (filtering thresholds)**

## Dataset
The project uses the **MovieLens 100K** dataset, which contains:
- **100,000** ratings
- **610 users**
- **9,724 movies**
- Ratings range from **1 to 5 stars**

## Installation
To run the project, install the required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib
