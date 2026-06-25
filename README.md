# Study Deep Learning

This repository consolidates my Deep Learning college coursework, exams, and textbook practice projects. It maps my study flow from building foundational neural networks to fine-tuning large language models (LLMs).

All documentation in this repository is maintained in English as the universal language.

---

## Study Flow and Directory Structure

The repository is organized into four main sections:

### 1. [01-hands-on-ml-geron/](01-hands-on-ml-geron/)
Practical exercises and notebooks from Chapters 1 to 19 of the book *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)* by Aurélien Géron. It covers:
*   **Chapters 1 - 9**: Classical machine learning algorithms, SVMs, decision trees, ensemble learning, and unsupervised learning.
*   **Chapters 10 - 19**: Deep learning foundations, custom models, image classification (CNN), sequence processing (RNN, LSTM, GRU), NLP (Transformers, Attention), Autoencoders, GANs, and Reinforcement Learning.

### 2. [02-tensorflow-in-action/](02-tensorflow-in-action/)
Practice notebooks from Chapters 1 to 11 of the book *TensorFlow in Action* by Thushan Ganegedara. It covers:
*   **Part 1 (Foundations)**: Basic computations (tf.Variable, tf.Tensor), Keras Sequential/Functional/Subclassing APIs, fully connected networks, CNNs, RNNs, and Transformer self-attention.
*   **Part 2 (Real-World Applications)**: Image classification, CNN improvements (dropout, data augmentation), image segmentation (DeepLabv3), and NLP sentiment analysis/language modeling.
*   **Part 3 (Sequence-to-Sequence)**: Encoder-decoder sequence translation from scratch.

### 3. [03-midterm-deep-learning/](03-midterm-deep-learning/)
Midterm exam portfolio covering three core machine learning tasks:
*   **Classification**: Credit card fraud detection on highly imbalanced transaction datasets.
*   **Regression**: Continuous numerical target prediction using high-dimensional data.
*   **Clustering**: Unsupervised credit card customer segmentation using K-Means and PCA.

### 4. [04-finalterm-deep-learning/](04-finalterm-deep-learning/)
Final exam tasks focusing on advanced deep learning and NLP tasks:
*   **Task 1**: Fine-tuning BERT models on AGNews, GoEmotions, and MNLI datasets.
*   **Task 2**: Fine-tuning T5 models on the SQuAD question-answering dataset.
*   **Task 3**: Parameter-efficient fine-tuning (PEFT/LoRA) of the Phi-2 model on the XSum summarization dataset.

---

## Setup and Requirements

To run the Jupyter notebooks (`.ipynb`) in this repository, you should configure a Python environment with the following dependencies:
- Python 3.8+
- Jupyter Notebook / JupyterLab
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-Learn
- TensorFlow, PyTorch
- Hugging Face Transformers, Datasets, PEFT, Accelerate