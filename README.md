# Credit Card Fraud Detection using Machine Learning

## Overview
This project focuses on building a Machine learning model to detect fraudulent credit card transactions.  
Since the dataset is **highly imbalanced** (only 0.17% fraud), I used **SMOTE** to balance the classes and applied multiple ML models to compare performance.

---

##  Problem Statement
Develop a binary classification model that predicts whether a transaction is:
- **Fraudulent** → `1`
- **Genuine** → `0`

I worked with anonymized transaction data, where features are derived from PCA (for confidentiality).

---

##  Dataset Details
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Records**: 284,807
- **Fraudulent Transactions**: 492 (~0.17%)
- **Features**:
  - `V1` to `V28`: Principal components (PCA)
  - `Time`, `Amount`: Original features
  - `Class`: Target variable (0 = normal, 1 = fraud)

---

## Tech Stack & Tools

**Languages & Libraries**:
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn, imblearn

**ML Models Used**:
- Naive Bayes
- Logistic Regression
- Decision Tree

**Other Tools**:
- `StandardScaler` for feature scaling  
- `SMOTE` to handle class imbalance  
- Correlation matrix for feature selection  
- Confusion Matrix & Classification Report for evaluation

---

##  Workflow

### ✅ Data Preprocessing
- No missing values 
- No feature dropped
- Scaled features using `StandardScaler`

###  Handling Class Imbalance
- Used **SMOTE** to oversample the minority class (fraud)
- Balanced dataset: 284,315 fraud vs 284,315 normal

###  Exploratory Data Analysis
- Visualized class distribution before & after balancing
- Used correlation matrix to find top fraud-related features:
  - Selected: `V10`, `V12`, `V14`, `V17`

---

##  Model Training & Results

| Model              | Accuracy (All Features) | Accuracy (Selected Features) |
|--------------------|-------------------------|-------------------------------|
| Naive Bayes        | 91.35%                  | 91.83%                        |
| Logistic Regression| 94.37%                  | 92.77%                        |
| Decision Tree      | **99.80%**              | 99.43%                        |

>  **Note**: While Decision Tree gave the highest accuracy, it may be overfitting due to the nature of the model. Logistic Regression was more stable.

---

##  Evaluation Metrics
- **Accuracy**
- **Precision / Recall / F1-Score** (important for fraud detection)
- **Confusion Matrix** (TP, TN, FP, FN)

---

## 🔍 Key Insights
- **Logistic Regression** was the most balanced across performance metrics
- **SMOTE** significantly boosted the ability to detect frauds
- **Correlation-based feature selection** reduced dimensionality without major performance loss
- High accuracy isn't everything — **recall** matters more for fraud detection!

---

## 💡 What I Learned
- How to deal with **imbalanced datasets** (using SMOTE, stratified splits, etc.)
- Importance of **precision vs recall** in real-world ML problems
- Comparing multiple models helps avoid overfitting traps
- You can extract strong insights even from **anonymized data** using EDA and correlation

---

##  How to Run This Project

1. **Clone the repo** or download the notebook
2. **Install the required libraries**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imblearn
