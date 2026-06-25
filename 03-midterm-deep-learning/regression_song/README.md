## README: Predictive Modeling for Continuous Variables

This repository contains a Jupyter Notebook (`regression_hit_song.ipynb`) that implements a complete solution for a **Regression** assignment, focusing on predictive modeling using a large, high-dimensional dataset.

---

## 🎯 Project Overview and Problem Statement

The objective of this project is to develop a machine learning model capable of accurately predicting a **continuous numerical Target Variable** based on a set of anonymized numerical features.

| Component | Description |
| :--- | :--- |
| **Task** | Regression (Predicting a continuous numerical value) |
| **Dataset Size** | 515,344 records with 91 columns (1 target + 90 features) |
| **Features** | 90 anonymized numerical features |
| **Target Variable** | Assumed to be the first column, renamed to `0` |
| **Models Used** | Ridge Regression (Baseline) and LightGBM Regressor |

---

## 🛠️ Methodology and Project Steps

The notebook follows a standard machine learning workflow:

1.  **Data Loading and Inspection:** The data is loaded from a CSV file (`midterm-regresi-dataset.csv`) mounted from Google Drive. The initial structure and data types are verified, noting that all features are numerical and there is no missing data in the loaded set.
2.  **Exploratory Data Analysis (EDA):** The distribution of the target variable and its relationship with a few key features is explored.
3.  **Data Preprocessing:**
    * The dataset is split into training and test sets.
    * Features are scaled using **StandardScaler** to normalize the data distribution, which is necessary for the Ridge Regression baseline model.
4.  **Model Training (Baseline & Advanced):**
    * A **Ridge Regression** model is trained as a simple baseline.
    * An advanced **LightGBM Regressor** is used for the final, high-performance prediction model.
5.  **Hyperparameter Tuning:** A grid search (`GridSearchCV`) is performed to tune the hyperparameters of the LightGBM Regressor to optimize performance on the validation set.
6.  **Model Evaluation:** The final, best-performing model is evaluated on the independent test set using multiple regression metrics.

---

## 📈 Key Results and Performance Metrics

The performance of the best-tuned **LightGBM Regressor** model on the **test set** is summarized by the following metrics:

| Metric | Formula | Result |
| :--- | :--- | :--- |
| **Coefficient of Determination ($\mathbf{R^2}$)** | $1 - \frac{SS_{res}}{SS_{tot}}$ | **0.3485** |
| **Mean Squared Error (MSE)** | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | **77.5375** |
| **Root Mean Squared Error (RMSE)** | $\sqrt{MSE}$ | **8.8055** |
| **Mean Absolute Error (MAE)** | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | **6.1299** |

**Conclusion:**
The model achieved an $R^2$ of **0.3485** and a RMSE of **8.8055** on the test set, indicating that approximately 35% of the variance in the target variable can be explained by the feature set using the trained model.

---

## ⚙️ How to Run the Notebook

1.  **Dependencies:** Ensure all required Python libraries are installed (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`).
2.  **Data:** The notebook requires access to the dataset stored at the path `/content/drive/MyDrive/midterm-regresi-dataset.csv`.
3.  **Execution:** Run the notebook cells sequentially in a Google Colab environment after mounting your Google Drive.
