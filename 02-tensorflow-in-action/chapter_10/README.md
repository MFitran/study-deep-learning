# Chapter 10: Autoencoders

This Jupyter Notebook (`notebook_10.ipynb`) contains the code and theoretical summaries from **Chapter 10 of "TensorFlow in Action" by Thushan Ganegedara**.

This notebook provides a comprehensive guide to **Autoencoders**, a powerful type of unsupervised neural network. The core idea is to learn a compressed, low-dimensional representation (latent vector) of data by training a model to reconstruct its own input.

An autoencoder is composed of two parts:
1.  **Encoder**: Compresses the input data into a smaller latent representation ($z$).
2.  **Decoder**: Takes the latent representation $z$ and attempts to reconstruct the original input data.

---

##  summary-of-contents

### 1. Simple (Vanilla) Autoencoder

* **Dataset:** MNIST (handwritten digits).
* **Architecture:** A basic autoencoder built using `Dense` layers.
* **Goal:** To demonstrate the fundamental concept. The model is trained to minimize the reconstruction loss (e.g., `binary_crossentropy` or `mse`) between the original images and the output of the decoder.

### 2. Denoising Autoencoder (DAE)

* **Dataset:** MNIST (with added Gaussian noise).
* **Concept:** A DAE is trained to learn more robust features by forcing it to recover the *original, clean image* from a *corrupted, noisy input*.
* **Goal:** This forces the model to learn the underlying structure of the data rather than just memorizing the input.

### 3. Convolutional Autoencoder (CAE)

* **Dataset:** Fashion-MNIST.
* **Architecture:** A more powerful autoencoder that uses convolutional layers, which are far more effective for image data.
* **Encoder:** Uses `Conv2D` and `MaxPooling2D` layers to progressively downsample the image into a latent vector.
* **Decoder:** Uses `Conv2DTranspose` and `UpSampling2D` layers to upsample the latent vector back into a full-sized image.

### 4. Image-to-Image Translation (U-Net as an Autoencoder)

* **Dataset:** `pix2pix` (facades dataset).
* **Concept:** This advanced example frames image-to-image translation (e.g., turning a segmentation mask into a photo) as an autoencoder task.
* **Architecture:** A **U-Net** is built.
* **U-Net:** This is a special type of encoder-decoder network that includes **skip connections**. Skip connections directly link layers from the encoder to corresponding layers in the decoder, which allows the model to pass fine-grained, low-level details (like edges) and significantly improves the sharpness and quality of the reconstructed image.

---

## 🛠️ Requirements

To run this notebook, you will need the following Python libraries:

* `tensorflow`
* `tensorflow-datasets` (tfds)
* `numpy`
* `matplotlib`

## 🚀 How to Use

You can run the `notebook_10.ipynb` file in any compatible Jupyter environment, such as Jupyter Lab, Jupyter Notebook, or Google Colab.
