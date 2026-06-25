# Chapter 11: Generative Adversarial Networks (GANs)

This Jupyter Notebook (`notebook_11.ipynb`) contains the code and theoretical summaries from **Chapter 11 of "TensorFlow in Action" by Thushan Ganegedara**.

This notebook provides an in-depth introduction to **Generative Adversarial Networks (GANs)**, a class of deep learning models where two neural networks compete against each other in a zero-sum game.

The two competing networks are:
1.  **The Generator (G):** A network that attempts to create new, synthetic data (e.g., images) from a random noise vector. Its goal is to produce data so realistic that it "fools" the discriminator.
2.  **The Discriminator (D):** A network that acts as a classifier. It is fed both real data (from the training set) and fake data (from the generator) and must learn to distinguish between the two.

The training process involves the Generator getting progressively better at creating fakes, while the Discriminator gets progressively better at spotting them, pushing the Generator to new levels of quality.

---

##  summary-of-contents

### 1. Simple GAN for MNIST

This first section builds a basic GAN from scratch using `Dense` layers to demonstrate the core training mechanics.

* **Dataset:** MNIST (handwritten digits).
* **Generator:** A simple `Sequential` model that takes a 100-dimensional noise vector as input and outputs a 784-pixel (28x28) image.
* **Discriminator:** A simple `Sequential` model that takes a 784-pixel image and classifies it as "real" (output 1) or "fake" (output 0).
* **Custom Training Loop:** A full custom training loop is implemented using `@tf.function` and `tf.GradientTape`. This loop explicitly shows the "adversarial" process:
    1.  The Discriminator is trained on a mixed batch of real and fake images.
    2.  The Generator is trained to produce images that cause the Discriminator to output "real" (1).

### 2. Deep Convolutional GAN (DCGAN)

This second section builds a much more powerful and modern GAN, the **DCGAN**, which uses convolutional layers to generate higher-quality, more complex images.

* **Dataset:** `celeb_a` (a large-scale dataset of celebrity faces), loaded via `tensorflow-datasets`.
* **Generator (DCGAN):** A `Sequential` model that uses `Conv2DTranspose` (upsampling) layers, `BatchNormalization`, and `LeakyReLU` to transform a 100-dimensional noise vector into a 64x64x3 color image.
* **Discriminator (DCGAN):** A `Sequential` model that uses `Conv2D` (downsampling) layers, `BatchNormalization`, `LeakyReLU`, and `Dropout` to classify a 64x64x3 image as real or fake.
* **Training:** The same custom training loop logic is applied, but with these new, more powerful convolutional models. A helper function is included to periodically save a grid of generated images, allowing you to visually monitor the Generator's progress as it learns to create realistic faces.

---

## 🛠️ Requirements

To run this notebook, you will need the following Python libraries:

* `tensorflow`
* `tensorflow-datasets` (tfds)
* `numpy`
* `matplotlib`

## 🚀 How to Use

You can run the `notebook_11.ipynb` file in any compatible Jupyter environment, such as Jupyter Lab, Jupyter Notebook, or Google Colab.
