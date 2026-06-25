# Chapter 6: Natural Language Processing with TensorFlow 2

This Jupyter Notebook (`notebook_06.ipynb`) contains the code and theoretical summaries from **Chapter 6 of *"TensorFlow in Action"* by Thushan Ganegedara**.

This notebook provides a complete walkthrough of building Natural Language Processing (NLP) models in TensorFlow, starting from raw text and building up to advanced recurrent architectures.

---

##  summary-of-contents

The notebook uses the **IMDb movie review dataset** as its primary example, with the goal of performing binary sentiment classification (classifying reviews as "positive" or "negative").

### 1. Text Preprocessing

This section demonstrates the modern approach to handling raw text in Keras.
* **`tf.keras.layers.TextVectorization`**: This layer is used to convert raw text strings into sequences of integer indices.
* **Vocabulary Building**: The `.adapt()` method is called on the layer to build a vocabulary from the training text data.

### 2. Word Embeddings

* **`tf.keras.layers.Embedding`**: This layer is introduced as the standard way to handle tokenized text.
* **Concept**: Instead of one-hot encoding, an `Embedding` layer learns a dense, low-dimensional vector representation for each word, capturing semantic relationships.

### 3. Recurrent Neural Networks (RNNs) for Sentiment Analysis

This section builds and trains three different RNN models for classification:

* **Model A: SimpleRNN**: A baseline model is built using the `tf.keras.layers.SimpleRNN` layer. This model demonstrates the basic RNN structure but suffers from the **vanishing gradient problem**, making it difficult to learn long-range dependencies in text.
* **Model B: LSTM (Long Short-Term Memory)**: As a solution to vanishing gradients, a new model is built using the `tf.keras.layers.LSTM` layer. LSTMs are a more advanced RNN cell designed to remember information over long sequences.
* **Model C: Stacked & Bidirectional LSTM**: A third, more powerful model is built using two advanced techniques:
    * **Stacked RNNs**: Using multiple RNN layers (by setting `return_sequences=True`).
    * **Bidirectional RNNs**: Using the `tf.keras.layers.Bidirectional` wrapper, which processes the text in both forward and backward directions to capture context from both ends.

---

## 🛠️ Requirements

To run this notebook, you will need the following Python libraries:
* `tensorflow`
* `numpy`

## 🚀 How to Use

You can run the `notebook_06.ipynb` file in any compatible Jupyter environment, such as Jupyter Lab, Jupyter Notebook, or Google Colab.
