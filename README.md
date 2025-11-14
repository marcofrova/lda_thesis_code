# Topic Modeling on ECB Speeches

Bachelor Thesis — Bocconi University, Milan  
**Supervisor:** Prof. Antonio Lijoi  
**Period:** May 2025 – July 2025

---

## Overview

This project builds an **unsupervised topic modeling pipeline** to analyze European Central Bank (ECB) speeches, aiming to uncover latent themes and sentiment trends across years of monetary policy communication.

The analysis is based on the **Latent Dirichlet Allocation (LDA)** algorithm, implemented in Python, and integrates preprocessing, tokenization, and visualization techniques to interpret textual data effectively.

---

## Objectives

- Extract latent topics from ECB speeches over multiple years.  
- Analyze **temporal evolution** of communication themes in monetary policy.  
- Evaluate **topic coherence** and optimize the number of topics.  
- Visualize topic structures and trends using **pyLDAvis** and custom plots.

---

## Pipeline Structure

1. **Data Preprocessing:**  
   Text cleaning, lemmatization, stopword removal, and tokenization.

2. **Vectorization:**  
   Conversion of text into document-term matrices via TF-IDF or Bag-of-Words.

3. **Topic Modeling:**  
   LDA training and hyperparameter tuning to maximize topic coherence.

4. **Visualization & Interpretation:**  
   - pyLDAvis interactive visualization  
   - Topic distribution over time  
   - Sentiment analysis on topic clusters

---

## Tools and Libraries

- Python 3.11  
- pandas, NumPy  
- scikit-learn  
- gensim  
- pyLDAvis  
- matplotlib / seaborn  
- NLTK / spaCy  

---

## Repository Structure
