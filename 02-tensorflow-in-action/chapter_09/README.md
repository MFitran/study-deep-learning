# Chapter 9: Unsupervised Learning with TensorFlow

This Jupyter Notebook (`notebook_09.ipynb`) contains the code and theoretical summaries from **Chapter 9 of "TensorFlow in Action" by Thushan Ganegedara**.

This notebook explores **unsupervised learning**, a branch of machine learning focused on finding hidden patterns and structures in data that does not have explicit labels. It covers three fundamental unsupervised tasks: dimensionality reduction, clustering, and recommender systems.

---

##  summary-of-contents

### 1. Dimensionality Reduction (PCA)

This section demonstrates how to reduce the number of features in a dataset while preserving as much information as possible using **Principal Component Analysis (PCA)**.

* **Concept:** PCA finds the "principal components" (directions of maximum variance) in the data and projects the data onto a new, lower-dimensional subspace defined by these components.
* **Dataset:** MNIST (handwritten digits).
* **Example:**
    * Uses `sklearn.decomposition.PCA` to compress the 784-dimensional MNIST images down to just 100 dimensions.
    * Visualizes the reconstructed images from the compressed data, showing that while some detail is lost, the digits remain recognizable.

### 2. Clustering (K-Means)

This section covers **K-Means**, a popular algorithm for partitioning data into a specified number ($K$) of distinct, non-overlapping clusters.

* **Concept:** The algorithm iteratively assigns data points to the nearest cluster "centroid" and then recalculates the centroid's position based on its assigned points.
* **Dataset:** A synthetic 2D dataset created with `sklearn.datasets.make_blobs`.
* **Example:**
    * Uses `sklearn.cluster.KMeans` to find 3 clusters in the synthetic data.
    * Visualizes the results with `matplotlib`, showing the original data points colored by their assigned cluster and marking the final cluster centers.

### 3. Recommender System (Matrix Factorization)

This section provides a more advanced, in-depth example, building a **collaborative filtering recommender system** from scratch using matrix factorization.

* **Concept:** **Matrix factorization** is a technique that learns low-dimensional latent "embedding" vectors for both users and items (e.g., movies). The dot product of a user's embedding and a movie's embedding is used to predict the rating that user would give to that movie.
* **Dataset:** `movielens/100k-ratings` from `tensorflow-datasets`.
* **Model:**
    * A Keras model is built using the **Functional API**.
    * The model takes `user_id` and `movie_id` as two separate inputs.
    * It contains two `Embedding` layers: one to learn user latent factors and one to learn movie latent factors.
    * A `Dot` layer is used to compute the dot product of the user and movie embeddings, producing a predicted rating.
* **Training:** The model is trained on `(user_id, movie_id, rating)` triplets, minimizing the Mean Squared Error (MSE) between its predicted ratings and the true ratings.

---

## 🛠️ Requirements

To run this notebook, you will need the following Python libraries:

* `tensorflow`
* `tensorflow-datasets` (tfds)
* `scikit-learn` (sklearn)
* `numpy`
* `matplotlib`

## 🚀 How to Use

You can run the `notebook_09.ipynb` file in any compatible Jupyter environment, such as Jupyter Lab, Jupyter Notebook, or Google Colab.
