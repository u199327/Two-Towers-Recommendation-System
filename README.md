# Two Towers Recommendation System
This repository contains the implementation of the Two Towers Recommendation System, a dual-tower model designed to provide scalable and efficient recommendations. The system uses two distinct recommendation approaches, weighted average and clustering-based methods, to optimize the trade-off between computational efficiency and recommendation accuracy.

## Overview
Recommendation systems are essential tools for personalizing user experiences by predicting preferences based on past interactions. The Two Towers model represents users and items in separate embedding spaces, enabling efficient and flexible recommendation strategies.

## Key Features:
- Weighted Average Method: Utilizes user-item relationships for quick and computationally efficient recommendations.
- Clustering-Based Method: Enhances accuracy by identifying patterns in the dataset using clustering techniques like K-Means and Gaussian Mixture Models.
- Optimized for large-scale datasets using just-in-time compilation with jax.jit.
- Explores solutions for cold-start scenarios, offering recommendations for new users with limited data.

## Dataset
The model is trained on the Anime Ratings Dataset from Kaggle, containing user-item interactions such as ratings and genres. The dataset was preprocessed to remove sparsity and ensure robust evaluations.

## Methods
- **Weighted Average Recommendations:**
  Calculates user embeddings based on the weighted average of rated items.
  Fast and suitable for scenarios with frequently changing data.

- **Clustering-Based Recommendations:**
  Groups items into clusters for efficient recommendation.
  Balances accuracy and computational efficiency by reducing search space during recommendations.

## Performance Metrics
The system is evaluated using standard metrics like:
- Precision@k
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)

These metrics highlight the model's trade-offs between speed and precision, providing insights for both seen and unseen user scenarios.

## Results
Clustering-based methods improve computational efficiency significantly by reducing the number of comparisons.
Weighted average methods excel in flexibility, especially for dynamic datasets.
Solutions for cold-start scenarios offer meaningful recommendations based on item popularity and embedding similarities.

This project was developed by :
- Marcel Fernández Serrano 
- Alejandro Pastor Rubio
- Xuezheng Zhang.
-
- Each contributor brought their expertise to the project, focusing on optimizing the recommendation system for scalability and accuracy.
