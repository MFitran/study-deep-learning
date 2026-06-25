# README: Credit Card Fraud Detection

## 💳 Fraud Transaction Assignment: Credit Card Fraud Detection

This repository contains a Jupyter Notebook (`classification_transaction_data.ipynb`) that implements an end-to-end **Machine Learning solution for Credit Card Fraud Detection**.

The goal of this project is to build a classification model capable of accurately predicting whether a given transaction is fraudulent (`isFraud = 1`) or legitimate (`isFraud = 0`).

---

## 🎯 Problem Statement and Challenge

The task is a **Binary Classification** problem. The primary challenge is the severe **class imbalance** present in the dataset, where fraudulent transactions constitute a very small minority (approximately 3.5% of the training data). This imbalance necessitates the use of specialized evaluation metrics.

| Component | Description |
| :--- | :--- |
| **Task** | Binary Classification (Fraud vs. Not Fraud) |
| **Dataset** | Large dataset with anonymized transaction and identity features (e.g., V-columns, C-columns). |
| **Target Distribution** | Non-Fraud: ~96.5%, Fraud: ~3.5%. |
| **Key Metrics** | **ROC-AUC**, **Precision**, and **Recall** (due to class imbalance). |

---

## 💻 Project Outline

The analysis and modeling process is structured into the following key steps:

1.  **Data Loading and Initial Inspection:** Load the transaction data from CSV files stored in Google Drive and inspect the data structure and target variable distribution.
2.  **Exploratory Data Analysis (EDA):** Analyze the target variable (to visualize class imbalance) and explore the distribution of key features, such as `TransactionAmt`.
3.  **Feature Engineering & Preprocessing (Simplified):** Due to the dataset's size and complexity (394 columns), a simplified preprocessing pipeline is implemented, focusing on a small, representative subset of features to demonstrate the handling of missing data and categorical variables.
4.  **Baseline Model Training:** Train a **Logistic Regression** model using the preprocessed data to establish a baseline performance.
5.  **Hyperparameter Tuning:** Improve the model performance by performing hyperparameter tuning using **`GridSearchCV`**.
6.  **Model Evaluation:** Evaluate the performance of the final, tuned model using the defined metrics (ROC-AUC, Confusion Matrix, and Classification Report) on a validation set.
7.  **Prediction on Test Data:** Use the best-trained model to predict the `isFraud` column for the `test_df` dataset.

---

## 📈 Key Results (Hyperparameter-Tuned Model)

The best model achieved the following performance on the validation set:

* **ROC-AUC Score:** **0.7603**
* **Fraud Class Recall (Sensitivity):** **0.65** (65% of actual fraudulent transactions were correctly identified)
* **Fraud Class Precision:** **0.09** (9% of transactions predicted as fraud were actually fraudulent)

This performance indicates the model is better than random (ROC-AUC > 0.5) and demonstrates a reasonable ability to detect fraud (Recall of 0.65), though the low precision highlights the trade-off in classifying an imbalanced dataset.

---

## ⚙️ How to Run the Notebook

1.  **Dependencies:** Ensure you have the necessary Python libraries installed (e.g., `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).
2.  **Data Access:** The notebook is set up to run in a Google Colab environment and requires the transaction CSV files (`train_transaction.csv` and `test_transaction.csv`) to be mounted from Google Drive.
3.  **Execution:** Open `classification_transaction_data.ipynb` and execute the cells sequentially.
