# 📚 Data Science & Engineering Labs: Spoken and Written Language Processing @ UPC

This repository showcases a collection of lab assignments and projects completed for the **Spoken and Written Language Processing** course within the **Bachelor's Degree in Data Science and Engineering** at **Universitat Politècnica de Catalunya (UPC)**. These projects offer hands-on experience in fundamental and advanced topics across Machine Learning, Natural Language Processing, Speech Recognition, and Deep Learning.

## 📌 Project Context

The labs in this repository are core components of the Spoken and Written Language Processing course, designed to provide practical exposure to the theoretical concepts of processing spoken and written language. Each assignment tackles a specific challenge, encouraging the application of various algorithms, models, and analytical techniques.

## 🧪 Labs Overview

### 1. 📝 Word Vectors: Training and Analysis (Lab 1)

* **Objective:** To implement, train, and analyze word vectors using the Continuous Bag-of-Words (CBOW) model from scratch.
* **Key Topics:** CBOW architecture, word embeddings, context-based prediction, cross-entropy loss, and visualizing semantic relationships in a vector space using PCA.
* **Focus:** Understanding the mechanics of distributed word representations and their utility in NLP.

### 2. 📖 Language Modeling (Lab 2)

* **Objective:** To study, develop, and compare different model architectures for non-causal language modeling, specifically focusing on predicting a masked (middle) word in a sequence.
* **Key Topics:** Feedforward Neural Networks for language modeling, scaling with Transformer Layers (Multi-Head Attention), domain adaptation strategies, shared input/output embeddings, and systematic hyperparameter optimization.
* **Focus:** Designing and optimizing neural network models to capture complex linguistic patterns and contextual dependencies.

### 3. 🗣️ Language Identification: Sentence Classification (Lab 3)

* **Objective:** To build and evaluate models capable of identifying the language of a given sentence.
* **Key Topics:** Hyperparameter tuning for classification models (embedding size, RNN hidden size, batch size, optimizers), combining pooling layers, concatenation, dropout layers, comparative analysis of different RNN architectures, and leveraging N-grams (bigrams, trigrams) and character counts as input features.
* **Focus:** Applying machine learning techniques to text classification tasks, with an emphasis on feature engineering and model architecture selection for optimal performance.

### 4. 🎤 Speech Recognition Using Dynamic Time Warping (Lab 4)

* **Objective:** To enhance the performance of a Dynamic Time Warping (DTW) based algorithm for isolated digit speech recognition.
* **Key Topics:** Principles of Dynamic Time Warping for sequence alignment, evaluation using Speak Error Rate (SER), advanced audio features like Cepstral Coefficients (MFCC), cepstral normalization, liftering, and incorporating first-order derivatives. Includes implementing backtracking for optimal alignment paths.
* **Focus:** Deepening understanding of classical speech recognition techniques and feature engineering for audio data.

### 5. 🩺 COVID-19 Detection from Coughs: Audio Classification (Lab 5)

* **Objective:** To develop and assess deep learning models for the detection of COVID-19 using audio recordings of coughs.
* **Key Topics:** Exploring VGG and HuBERT-based architectures for audio classification, fine-tuning hyperparameters, utilizing pre-trained speech embeddings (HuBERT, Wav2Vec, PASE, VGGISH) through transfer learning, implementing weighted average of hidden states, and employing techniques like gradient accumulation and K-fold cross-validation for robust training.
* **Focus:** Applying cutting-edge deep learning models and audio processing techniques to a real-world biomedical classification challenge.

## 🛠️ Common Tools & Technologies

These projects are primarily implemented in **Python**, leveraging a powerful ecosystem of libraries for data science and machine learning:

* **Deep Learning Frameworks:** `PyTorch` (or `TensorFlow`) for building and training neural networks.
* **Data Manipulation & Analysis:** `NumPy` and `Pandas`.
* **Machine Learning Utilities:** `Scikit-learn` for various algorithms, preprocessing, and model evaluation.
* **Specialized Libraries:** `imbalanced-learn` (for handling imbalanced datasets), and potentially domain-specific libraries for audio feature extraction or processing.
