# Chapter 4: Model Training in TensorFlow 2

This Jupyter Notebook (`notebook_04.ipynb`) contains the code and theoretical summaries from **Chapter 4 of "TensorFlow in Action" by Thushan Ganegedara**.

This notebook provides a deep dive into the two primary methods for training models in TensorFlow 2: the high-level `model.fit()` API and the low-level, fully-customizable Custom Training Loop (CTL). It also covers how to build your own custom loss functions and metrics.

The **Iris dataset** is used for all classification examples.

---

##  summary-of-contents

### 1. Built-in Training (`model.fit`)

This section demonstrates the standard, high-level method for training Keras models. It's the simplest and most common approach.

* **Model:** A `Sequential` Keras model (Model A) is built for the Iris dataset.
* **Compile:** The `model.compile()` method is used to configure the training process, specifying:
    * `optimizer` (e.g., 'adam')
    * `loss` (e.g., 'categorical_crossentropy')
    * `metrics` (e.g., ['acc'])
* **Fit:** The `model.fit()` method is called to train the model with a single line of code, handling the training loop, validation, and metric reporting automatically.

### 2. Custom Training Loops (CTLs)

This section explains how to build a training loop from scratch. This method provides maximum flexibility and control over the training process, which is essential for research and non-standard model architectures.

* **Core Component: `tf.GradientTape`**
    * Explains how `tf.GradientTape` is used as a "tape recorder" to watch computations involving trainable variables.
    * It automatically calculates the gradients (derivatives) of a target (like loss) with respect to those variables using `tape.gradient()`.
* **Manual Gradient Application**
    * Demonstrates how to use an optimizer (e.g., `tf.keras.optimizers.Adam`) to manually apply the calculated gradients to the model's weights using `optimizer.apply_gradients()`.
* **Full CTL Implementation**
    * Rebuilds Model A and trains it using a manual `for` loop.
    * This loop explicitly shows every step of training:
        1.  Start a `tf.GradientTape`.
        2.  Perform a forward pass (get predictions).
        3.  Calculate the loss.
        4.  Calculate gradients of the loss w.r.t. model weights.
        5.  Apply gradients to update the weights.
        6.  Manually update and print metrics.

### 3. Custom Components (Losses and Metrics)

This section shows how to create your own custom loss functions and metrics by sub-classing the base Keras classes. This allows for implementing logic not found in the standard library.

* **Custom Loss Function**
    * A `CustomCategoricalCrossentropy` class is built by inheriting from `tf.keras.losses.Loss`.
    * It overrides the `__init__()` and `call
