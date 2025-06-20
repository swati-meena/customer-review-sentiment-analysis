<h1 align="center">🍽️ Restaurant Review Sentiment Classifier</h1>
<p align="center">
  A complete NLP pipeline to classify restaurant reviews as Positive ✅ or Negative ❌ using multiple ML algorithms and vectorization techniques.
</p>

---

## 📌 Overview

This project demonstrates the application of various Machine Learning algorithms on restaurant review data using two key feature extraction methods:

- ✅ `CountVectorizer`
- ✨ `TfidfVectorizer`

We compare how different algorithms perform with each vectorization technique and present the accuracy scores for each.

---

## 🧠 Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLTK-darkgreen?style=for-the-badge&logo=nltk&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Tfidf-TextVector-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ML-Models-yellow?style=for-the-badge"/>
</p>

---

## 🧹 Preprocessing Steps

- Remove punctuation and special characters
- Convert to lowercase
- Remove stopwords
- Apply stemming using Porter Stemmer
- Store cleaned text in a `corpus`

---

## 📊 Algorithms Tried

| Algorithm               | Accuracy (TF-IDF) | Accuracy (CountVectorizer) |
|-------------------------|------------------|-----------------------------|
| Logistic Regression     | `76.0%`          | `67.5%`                     |
| Decision Tree (Entropy) | `71.5%`          | `63.4%`                     |
| Random Forest           | `72.0%` ✅       | `68.0%`                     |
| K-Nearest Neighbors     | `67.5%`          | `67.9%`                     |
| LightGBM                | '64.5%           | `65.5%`                     |

> 🎯 **Best Model:** `Logistic Regression + TF-IDF` = `76%`

---

## 🛠 How to Run

1. Clone the repository:
```bash
git clone https://github.com/swati-meena/restaurant-review-classifier.git
cd restaurant-review-classifier
