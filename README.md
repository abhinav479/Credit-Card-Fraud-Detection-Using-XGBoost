# Credit Card Fraud Detection Using Machine Learning

## Project Overview
This project focuses on developing a robust machine learning model to detect fraudulent credit card transactions from a highly imbalanced dataset. The goal is to accurately identify fraudulent transactions while minimizing false positives, using advanced preprocessing and diverse classification models.

***

## Libraries and Tools Used

- **Python 3.x** — core programming language  
- **Pandas** — data manipulation and analysis  
- **NumPy** — numerical operations  
- **Matplotlib & Seaborn** — data visualization  
- **Scikit-learn** — machine learning utilities, model selection, metrics  
- **Imbalanced-learn (imblearn)** — dealing with class imbalance using SMOTE  
- **XGBoost** — gradient boosting library for high-performance classification  
- **LightGBM** — fast, scalable gradient boosting framework  
- **Optuna** — hyperparameter optimization framework for efficient tuning  

***

## Dataset Description
- Dataset with over 250,000 credit card transactions with features mostly anonymized by PCA.
- Highly imbalanced: only ~0.17% transactions are fraudulent.
- Key features include anonymized V1–V28 PCA components, `Amount`, and `Time`.
- Target variable is binary: 0 (non-fraud), 1 (fraud).

***

## Project Workflow

### Data Preprocessing
- Removed duplicate records to ensure data quality.
- Handled missing values if any.
- Used **SMOTE** to oversample the minority fraud class, balancing the training data.
- Split the dataset into train, validation, and test sets, preserving class distribution.

### Exploratory Data Analysis (EDA)
- Examined class distribution and feature correlations.
- Visualized feature statistics and imbalances.

### Models Trained
- **XGBoost** — gradient boosted trees with hyperparameter tuning and early stopping.
- **Random Forest** — ensemble decision trees, tuned for fraud detection.
- **LightGBM** — gradient boosting with tuned parameters.
- **Logistic Regression** — baseline linear model for comparison.

### Hyperparameter Tuning
- Used **Optuna** for efficient Bayesian optimization of XGBoost hyperparameters.
- Focused on **maximizing recall** to catch the highest fraud rate.
- Tuned parameters like `max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, `learning_rate`, and regularization.

### Model Evaluation
- Evaluated using precision, recall, and F1-score with emphasis on fraud class.
- Also used ROC-AUC and PR-AUC for ranking capability and imbalanced data performance.
- Compared all models side-by-side using these metrics.

***

## Results Summary

| Model               | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) | ROC-AUC | PR-AUC |
|---------------------|-------------------|----------------|------------------|---------|--------|
| XGBoost (Tuned)     | 0.52              | 0.82           | 0.64             | 0.961600    | 0.813800 |
| XGBoost (Default)   | 0.50              | 0.80           | 0.62             | 0.954400   |0.803100  |
| Random Forest       | 0.93              | 0.75           | 0.83             | 0.943800    | 0.812100  |
| LightGBM            | 0.33              | 0.80           | 0.47             | 0.950600   |0.775100   |
| Logistic Regression | 0.05              | 0.87           | 0.10             | 0.964200  | 0.683100|

***

## Key Conclusions
- XGBoost with tuning offers the best overall balance for fraud detection.
- Random Forest shows high precision but lower recall compared to XGBoost.
- Logistic Regression, while having high recall, is impractical due to many false positives.
- PR-AUC is a better metric for this imbalanced use case than accuracy.

***

## How to Run
- Preprocessing scripts to clean and balance the dataset.
- Model training scripts using XGBoost with early stopping.
- Optuna tuning for XGBoost.
- Evaluation scripts printing detailed classification reports.

