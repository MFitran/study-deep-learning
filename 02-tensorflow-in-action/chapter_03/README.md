# Chapter 3: Keras and Data Retrieval in TensorFlow 2

This Jupyter Notebook (`notebook_03.ipynb`) contains the code and theoretical summaries from **Chapter 3 of "TensorFlow in Action" by Thushan Ganegedara**.

The notebook is divided into two main sections, covering the essential skills for building and feeding data to models in TensorFlow.

---

##  summary-of-contents

### 1. Keras Model-Building APIs

This section demonstrates how to build neural networks using the three distinct APIs provided by Keras, using the **Iris dataset** for classification.

* **Sequential API:**
    * **Model A** is built as a simple, linear stack of `Dense` layers.
    * This is the most straightforward approach, ideal for models with a single input and single output.

* **Functional API:**
    * **Model B** is built to demonstrate a more complex, multi-input architecture.
    * It takes two inputs:
        1.  The raw Iris features (4 values).
        2.  The first two Principal Components (PCA) of the features (2 values).
    * The outputs of these two input branches are concatenated before being passed to the final classification layers.

* **Sub-classing API:**
    * **Model C** demonstrates the most flexible and customizable approach by creating a custom layer.
    * A new layer, `MulBiasDense`, is defined by sub-classing `tf.keras.layers.Layer`.
    * This custom layer overrides the `__init__`, `build`, and `call` methods to add a *multiplicative bias* ($b_{mul}$) in addition to the standard additive bias ($b$), implementing the formula: $h = \alpha([xW + b] \times b_{mul})$.

### 2. Data Retrieval in TensorFlow

This section demonstrates three different methods for creating efficient data input pipelines to feed models.

* **The `tf.data` API:**
    * The recommended method for building high-performance, complex, and scalable data pipelines.
    * The example pipeline reads image file paths from a CSV, uses `.map()` to load and preprocess images, zips them with their labels, and then shuffles and batches the data.

* **Keras DataGenerators:**
    * A simpler, high-level utility, particularly for image data.
    * The example uses `ImageDataGenerator.flow_from_dataframe()` to read image file paths and labels directly from a `pandas.DataFrame` and generate batches of data.

* **The `tensorflow-datasets` (TFDS) Package:**
    * The easiest way to access hundreds of common, pre-processed datasets.
    * The example shows how to load the **CIFAR-10** dataset with a single line (`tfds.load("cifar10")`) and then use the `.map()` method to apply normalization and one-hot encoding.

---

## 🛠️ Requirements

To run this notebook, you will need the following Python libraries:

* `tensorflow` (and `tf.keras`)
* `pandas`
* `numpy`
* `requests`
* `scikit-learn` (for PCA)
* `tensorflow-datasets`
* `pillow` (PIL)

## 🚀 How to Use

You can run the `notebook_03.ipynb` file in any compatible Jupyter environment, such as Jupyter Lab, Jupyter Notebook, or Google Colab.
