## README: Credit Card Customer Segmentation

This repository contains a Jupyter Notebook (`clustering_customer_behavior.ipynb`) that implements a complete **Unsupervised Machine Learning** solution for **Credit Card Customer Segmentation**.

---

## 🎯 Project Overview and Problem Statement

The core objective is to apply clustering techniques to a credit card transaction dataset to discover distinct and meaningful groups of customers based on their spending and payment behavior.

The discovered customer segments can be leveraged by the bank for business strategies such as:
* Targeted marketing campaigns.
* Customized product and service offerings.
* Improved risk management.

| Component | Description |
| :--- | :--- |
| **Task** | Unsupervised Learning: Clustering (Customer Segmentation) |
| **Dataset** | Credit Card Dataset (`clusteringmidterm.csv`) featuring metrics like `BALANCE`, `PURCHASES`, `CREDIT_LIMIT`, and `PAYMENTS`. |
| **Algorithm** | $\mathbf{K}$-Means Clustering. |
| **Evaluation** | Elbow Method for optimal $\mathbf{K}$ determination, Silhouette Score for validation, and Cluster Profiling for interpretation. |

---

## 🛠️ Methodology and Project Steps

The notebook follows a standard clustering workflow:

1.  **Data Loading and Initial Inspection:** Load the `clusteringmidterm.csv` file and check for data types and missing values.
2.  **Data Preprocessing:**
    * Handle Missing Values: Impute missing values in `MINIMUM_PAYMENTS` and `CREDIT_LIMIT` using the **median** strategy.
    * Feature Selection: Drop the non-feature identifier column, `CUST_ID`.
    * Feature Scaling: Scale all features using **StandardScaler** to ensure equal contribution to the distance calculation in $\mathbf{K}$-Means.
3.  **Optimal Cluster Determination:** The **Elbow Method** is applied to the WCSS (Within-Cluster Sum of Squares) to visually estimate the optimal number of clusters ($\mathbf{K}$).
4.  **Model Training:** The $\mathbf{K}$-Means model is trained using the determined $\mathbf{K}$ (e.g., $K=4$).
5.  **Evaluation and Interpretation:**
    * **Visualization:** **Principal Component Analysis (PCA)** is used to reduce the 17-dimensional data to 2 principal components (PC1 and PC2) for a visual scatter plot of the clusters.
    * **Cluster Profiling:** The mean values of key original features (`BALANCE`, `PURCHASES`, `CREDIT_LIMIT`, etc.) are analyzed for each cluster to characterize the customer segments.

---

## 📈 Key Results and Interpretation

The $\mathbf{K}$-Means model with the chosen optimal $\mathbf{K}$ (e.g., $K=4$) successfully segments the customer base.

### Example Cluster Profiles (Based on Mean Feature Values):

The notebook's interpretation suggests the following segment characteristics, derived from analyzing feature means:

* **Cluster 0:** Characterized by high balance, high purchases, and high payments (interpreted as **Affluent Users**).
* **Cluster 1:** High cash advance usage and low purchases (interpreted as **Cash-focused Users**).
* **Cluster 2:** Low values across most features (interpreted as **Inconsistent/Small Spenders**).
* **Cluster 3:** High credit limit and high proportion of full payments (interpreted as **Responsible, High-Limit Users**).

The Silhouette Score is also calculated to provide an internal measure of cluster quality.

---

## ⚙️ How to Run the Notebook

1.  **Dependencies:** Ensure necessary Python libraries are installed, including `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` (`StandardScaler`, `SimpleImputer`, `KMeans`, `PCA`).
2.  **Data:** The notebook requires the dataset file named `clusteringmidterm.csv` to be accessible in the execution environment.
3.  **Execution:** Run the cells sequentially in your Jupyter or Colab environment. The results include a plot of the Elbow Method and a PCA-based visualization of the final clusters, which are saved as PNG files.
