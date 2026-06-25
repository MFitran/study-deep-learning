# Chapter 2: First Principles of TensorFlow 2

This Jupyter Notebook (`notebook_02.ipynb`) contains the code and theoretical summaries from **Chapter 2 of "TensorFlow in Action" by Thushan Ganegedara**.

The notebook provides a foundational introduction to TensorFlow 2, focusing on its core component (Tensors) and culminating in a complete, from-scratch implementation of a linear regression model.

---

##  summary-of-contents

### 1. Tensors

This section introduces `tf.Tensor`, the fundamental data structure in TensorFlow.
* **Definition:** Tensors are multi-dimensional arrays, similar to NumPy arrays.
* **Core Types:**
    * `tf.constant`: Immutable (non-trainable) tensors.
    * `tf.Variable`: Mutable (trainable) tensors, used for model weights.
* **Creation:** Demonstrates creating tensors using:
    * `tf.constant()`
    * `tf.Variable()`
    * `tf.zeros()` and `tf.ones()`
    * `tf.random.normal()` and `tf.random.uniform()`
* **Attributes:** Covers inspecting tensors via `.shape`, `.dtype`, and converting to NumPy arrays with `.numpy()`.

### 2. Operations on Tensors

This section covers the basic mathematical and manipulative operations that can be performed on tensors.
* **Arithmetic:** Basic element-wise operations (addition, subtraction, multiplication).
* **Matrix Operations:** Demonstrates matrix multiplication using `tf.matmul()`.
* **Manipulation:**
    * `tf.reshape`: Changing the shape of a tensor.
    * `tf.cast`: Changing the data type of a tensor.
    * `tf.transpose`: Swapping tensor dimensions.
    * `tf.concat`: Joining multiple tensors along an axis.

### 3. Linear Regression from Scratch

This is the main practical example in the notebook. It demonstrates how to build and train a simple linear regression model to predict housing prices using the **Boston Housing dataset**.

The implementation is broken down into key steps:

* **Data Preparation:** Loading the dataset (using `pandas` and `tensorflow.keras.datasets`) and preprocessing it (normalization, creating `tf.data.Dataset` pipelines).
* **Model Definition:** Defining the linear model as `y = Wx + b`, where `W` (weights) and `b` (bias) are `tf.Variable` objects.
* **Loss Function:** Defining the Mean Squared Error (MSE) loss function to measure model error.
* **Gradient Calculation:**
    * This is the core of "from-scratch" training.
    * It shows how to use `tf.GradientTape` to automatically record operations and compute the gradients (derivatives) of the loss with respect to the trainable variables (`W` and `b`).
* **Training Loop:**
    * A manual training loop is implemented.
    * In each epoch, the loop:
        1.  Calculates the model's predictions.
        2.  Calculates the loss (MSE) based on the predictions.
        3.  Uses the `GradientTape` to get the gradients.
        4.  Updates the `W` and `b` variables by "stepping" them in the opposite direction of the gradient, scaled by a learning rate (Stochastic Gradient Descent).

---

## 🛠️ Requirements

To run this notebook, you will need the following Python libraries:

* `tensorflow`
* `pandas`
* `numpy`

## 🚀 How to Use

You can run the `notebook_02.ipynb` file in any compatible Jupyter environment, such as Jupyter Lab, Jupyter Notebook, or Google Colab.
