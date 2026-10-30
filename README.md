# Netflix Content Analytics & Recommendation Engine
**Auspify Technologies - Machine Learning Internship Program**

This repository contains the complete implementation of 6 machine learning tasks analyzing the Netflix content dataset, completed as part of the Auspify Technologies 4-Week Practical Learning Program. 

## Project Overview
The project applies various machine learning techniques—ranging from Natural Language Processing (NLP) to unsupervised clustering and time-series forecasting—to extract insights and build predictive models for streaming content.

### Tasks Completed
1. **Content Recommendation System (NLP):** A content-based filtering system using TF-IDF and Cosine Similarity to recommend similar titles.
2. **Content Type Prediction (Binary Classification):** A Random Forest model predicting whether a title is a Movie or TV Show with ~99% accuracy.
3. **Audience Rating Classification (Multi-Class):** A decision tree-based model predicting target audience ratings based on content features.
4. **Content Segmentation (Clustering):** A K-Means clustering implementation using PCA dimensionality reduction to group content into 5 distinct genre clusters.
5. **Trend Forecasting (Time-Series):** A Linear Regression model predicting future Netflix content release volumes based on historical platform data.
6. **Success Analytics Engine (MLOps):** An end-to-end scikit-learn Pipeline comparing Random Forest and Logistic Regression classifiers, with the winning model serialized via joblib for production deployment.

## Tech Stack
* **Language:** Python 3
* **Libraries:** pandas, scikit-learn, numpy, matplotlib, seaborn
* **Techniques:** TF-IDF, K-Means, Random Forest, Logistic Regression, Linear Regression, PCA, Pipeline creation

## Setup & Execution
1. Clone this repository.
2. Ensure the `Dataset.csv` file is in the root directory.
3. Install dependencies: `pip install pandas scikit-learn numpy matplotlib seaborn tabulate`
4. Run individual task scripts (e.g., `python task1_recommendation.py`).
