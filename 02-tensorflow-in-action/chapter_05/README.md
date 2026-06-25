# Chapter 5: Computer Vision with CNNs

This Jupyter Notebook (`notebook_05.ipynb`) contains the code and theoretical summaries from **Chapter 5 of "TensorFlow in Action" by Thushan Ganegedara**.

This notebook provides a comprehensive guide to building Convolutional Neural Networks (CNNs) for computer vision tasks. It starts with the basic components and progressively builds to advanced, high-performance techniques like data augmentation and transfer learning.

---

##  summary-of-contents

### 1. The Building Blocks of a CNN

This section introduces the fundamental layers that form a Convolutional Neural Network:
* **`Conv2D` (Convolutional Layers):** Explains how these layers use filters (kernels) to perform convolutions, which slide over the input image to extract key features like edges, textures, and shapes.
* **`Pooling` (Subsampling Layers):** Covers `MaxPooling2D` and `AveragePooling2D`. These layers are used to reduce the spatial dimensions (downsample) of the feature maps, which makes the model more efficient and robust to variations in where features appear.

### 2. Building a CNN Classifier (Malaria Detection)

This section provides a practical, from-scratch implementation of a CNN to solve a binary classification problem.
* **Dataset:** Uses the `tensorflow-datasets` (tfds) library to load the **Malaria dataset**.
* **Goal:** Classify cell images as either **'parasitized'** or **'uninfected'**.
* **Preprocessing:** A `format_data` function is created to resize all images to a uniform (125, 125) and normalize pixel values to be between 0 and 1.
* **Model:** A `Sequential` model is built by stacking `Conv2D` and `MaxPool2D` layers, followed by a `Flatten` layer and `Dense` layers for the final classification.
* **Training:** The model is compiled using `binary_crossentropy` loss and trained using `ModelCheckpoint` and `EarlyStopping` callbacks to save the best-performing model and prevent overfitting.
* **Evaluation:** The trained model's performance is measured on the unseen test set.

### 3. Data Augmentation

This section explains how to improve the model's robustness and prevent overfitting by artificially expanding the training dataset.
* **Concept:** Data augmentation creates modified versions of the training images (e.g., flipping, rotating, zooming). This helps the model learn to ignore irrelevant variations and focus on the true features.
* **Implementation:** Demonstrates the modern TensorFlow 2 approach by adding Keras preprocessing layers (`RandomFlip`, `RandomRotation`, `RandomZoom`) **directly into the model architecture**. This allows the augmentations to be performed efficiently on-the-fly on the GPU during training.

### 4. Transfer Learning

This section introduces one of the most powerful techniques in modern computer vision: using a pre-trained model.
* **Concept:** Transfer learning leverages a model (like `MobileNetV2`) that has already been trained on a massive dataset (like ImageNet) and adapts it for a new, specific task.
* **Implementation:**
    1.  **Base Model:** Loads the `MobileNetV2` model from `tf.keras.applications`, keeping its pre-trained weights but removing its original classification head (`include_top=False`).
    2.  **Freezing:** The layers of the `MobileNetV2` base are frozen (`base_model.trainable = False`) to preserve their powerful, learned features.
    3.  **New Head:** A new, small classification "head" (consisting of `GlobalAveragePooling2D` and `Dense` layers) is added on top of the frozen base.
    4.  **Training:** Only the new head is trained on the Malaria dataset. This process is extremely fast and achieves very high accuracy, as it builds upon a strong foundation of pre-learned features.

---

## 🛠️ Requirements

To run this notebook, you will need the following Python libraries:

* `tensorflow`
* `tensorflow-datasets` (tfds)
* `numpy`

## 🚀 How to Use

You can run the `notebook_05.ipynb` file in any compatible Jupyter environment, such as Jupyter Lab, Jupyter Notebook, or Google Colab.
