## README: Consolidated Machine Learning Portfolio

This repository contains a collection of three independent machine learning projects demonstrating proficiency across supervised (Classification and Regression) and unsupervised (Clustering) learning tasks. Each project addresses a distinct business problem using Python and the `scikit-learn` ecosystem.

---

## 💻 Project Summary

| Project Notebook | ML Task | Core Problem | Key Algorithm(s) | Key Result/Metric |
| :--- | :--- | :--- | :--- | :--- |
| `classification_transaction_data.ipynb` | **Classification** | Credit Card Fraud Detection (Imbalanced) | Logistic Regression, Hyperparameter Tuning | **ROC-AUC: 0.7603**, Recall: 0.65 |
| `regression_hit_song.ipynb` | **Regression** | Predictive Modeling for Continuous Target | Ridge Regression (Baseline), LightGBM Regressor | $\mathbf{R^2}$: **0.3485**, RMSE: 8.8055 |
| `clustering_customer_behavior.ipynb` | **Clustering** | Credit Card Customer Segmentation | $\mathbf{K}$-Means Clustering, PCA | **4 Distinct Customer Segments** identified |

---

## 1. Credit Card Fraud Detection (Classification)

This project implemented an end-to-end Machine Learning solution for the **Credit Card Fraud Detection** problem.

* **Task:** Binary Classification (Fraud vs. Not Fraud).
* **Challenge:** The dataset presents a severe **class imbalance**, with fraudulent transactions being a small minority (approx. 3.5%).
* **Model & Process:** A **Logistic Regression** model was trained after simplified feature engineering and preprocessing. Hyperparameter tuning was performed using `GridSearchCV`.
* **Key Results (Validation Set):**
    * **ROC-AUC Score:** **0.7603**
    * **Fraud Class Recall (Sensitivity):** **0.65** (65% of actual fraudulent transactions were correctly identified)

## 2. Predictive Modeling for Continuous Variables (Regression)

This notebook focused on building a model to accurately predict a continuous numerical **Target Variable** using a large, high-dimensional dataset (515,344 records, 90 features).

* **Task:** Regression.
* **Data Preprocessing:** Features were scaled using **StandardScaler** prior to modeling.
* **Model & Process:** A **Ridge Regression** model was used as a baseline, followed by a high-performance **LightGBM Regressor**. Hyperparameter tuning was applied to the LightGBM model.
* **Key Results (Test Set):**
    * **Coefficient of Determination ($\mathbf{R^2}$):** **0.3485**
    * **Root Mean Squared Error (RMSE):** **8.8055**
    * **Mean Absolute Error (MAE):** **6.1299**

## 3. Credit Card Customer Segmentation (Clustering)

This project implemented an **Unsupervised Learning** solution for customer segmentation using the provided credit card dataset (`clusteringmidterm.csv`).

* **Task:** Clustering (Customer Segmentation).
* **Algorithm:** $\mathbf{K}$-Means Clustering.
* **Process:**
    * Missing values in `MINIMUM_PAYMENTS` and `CREDIT_LIMIT` were imputed using the median.
    * The **Elbow Method** was used to determine the optimal number of clusters ($\mathbf{K}$).
    * **PCA** was used for dimensionality reduction to visualize the final clusters.
* **Key Findings (Cluster Profiles):** The analysis successfully identified **4 distinct customer segments**, characterized by their mean feature values. Examples include:
    * **Affluent Users:** High Balance, High Purchases, High Payments.
    * **Cash-focused Users:** High Cash Advance, Low Purchases.
    * **Responsible Users:** High Credit Limit, High Proportion of Full Payments.
