# Chapter 7: A Deeper Look into Keras

This Jupyter Notebook (`nb_07 (1).ipynb`) contains the code and theoretical summaries from **Chapter 7 of "TensorFlow in Action" by Thushan Ganegedara**.

This notebook explores advanced Keras features that are essential for managing, monitoring, and optimizing the model training process. It covers callbacks for automating tasks, TensorBoard for in-depth visualization, and Keras Tuner for hyperparameter optimization.

---

##  summary-of-contents

### 1. Callbacks in Keras

Callbacks are utilities that can be applied at different stages of the `model.fit()` training process (e.g., at the start of an epoch, at the end of a batch). This section demonstrates the most important ones using a simple MNIST model:

* **`ModelCheckpoint`**: Automatically saves the model during training. The example shows how to save only the **best-performing model** based on validation accuracy (`save_best_only=True`).
* **`EarlyStopping`**: Monitors a metric (like `val_loss`) and stops the training process early if the metric stops improving, which helps **prevent overfitting**.
* **`TensorBoard`**: Logs metrics, graphs, and other information so it can be visualized in the TensorBoard interface.
* **`LearningRateScheduler`**: Allows you to define a custom schedule for changing the learning rate during training (e.g., decreasing it over time).

### 2. Advanced Visualization with TensorBoard

This section goes deeper into TensorBoard, moving beyond simple metric plots.

* **Visualizing the Model Graph**: Shows how TensorBoard can be used to inspect the model's architecture graph.
* **Embedding Projector**: A detailed example using the **Fashion MNIST dataset** demonstrates how to use the TensorBoard Embedding Projector. This powerful tool is used to visualize the high-dimensional embedding vectors learned by the model, allowing you to see how the model "learns" to cluster and separate the different classes of images.

### 3. Hyperparameter Tuning with Keras Tuner

This section introduces `keras-tuner`, the official Keras library for finding the best set of hyperparameters for your model.

* **Model-Building Function**: Demonstrates how to define a `build_model(hp)` function where `hp` is a Keras Tuner object used to define searchable hyperparameters (e.g., `hp.Choice()` for learning rate, `hp.Int()` for the number of units in a layer).
* **Tuner Instance**: Shows how to instantiate a tuner, such as `RandomSearch`, specifying the objective to optimize (e.g., `val_accuracy`).
* **Running the Search**: The `tuner.search()` method is used to automatically train multiple versions of the model to find the best hyperparameter combination.
* **Retrieving the Best Model**: After the search is complete, `tuner.get_best_models()` is used to retrieve the fully trained, optimized model.

---

## 🛠️ Requirements

To run this notebook, you will need the following Python libraries:

* `tensorflow`
* `keras-tuner`
* `numpy`
* `os`
* `datetime`

## 🚀 How to Use

You can run the `nb_07 (1).ipynb` file in any compatible Jupyter environment, such as Jupyter Lab, Jupyter Notebook, or Google Colab.
