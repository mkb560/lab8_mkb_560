# Lab 8: Comparative Analysis of Doc2Vec and Word2Vec

**Group 10** | Mingtao Ding, Ke Wu, Yi-Hsien Lou

## 📌 Project Overview
This repository contains a comparative evaluation of **Doc2Vec** and **Word2Vec** embedding models for document clustering. Using a scraped dataset of Reddit posts focused on **Information Security**, this lab contrasts the ability of document-level (Doc2Vec) and word-level (Word2Vec) architectures to capture and partition semantic meaning.

## ⚙️ Methodology
* **Doc2Vec:** Generates native document-level embeddings (100-dim) using `gensim`. Hyperparameters were optimized by maximizing the Silhouette Cosine score.
* **Word2Vec (Bin-Frequency):** Generates word-level embeddings, clusters them into $K$ semantic bins ($K=50, 100, 200$) using K-Means, and represents each document as a normalized $K$-dimensional bin-frequency vector.

## 📊 Evaluation Framework
Models were evaluated across three dimensions:
1. **Quantitative:** Silhouette Cosine score (measuring spatial cohesion and separation).
2. **Structural:** Cluster Size Distribution to check for imbalance.
3. **Semantic Coherence:** Manual evaluation of intra-cluster semantic consistency (scored 1-5).

## 🏆 Key Findings
* **Silhouette Scores:** Doc2Vec achieved a significantly higher score (0.7483 at $K=50$) compared to Word2Vec's near-zero peak (0.0097 at $K=100$), which mathematically indicated severe cluster overlap in the Word2Vec space.
* **The "Mega-Cluster" Effect:** Both models suffered from extreme structural imbalance. For example, Doc2Vec's largest cluster absorbed over 4,700 posts, leaving remaining clusters as highly marginalized outliers (as few as 3 posts). 
* **Manual Evaluation:** Semantic coherence scored low (2.0–2.8/5). Evaluators found posts across different clusters to be semantically indistinguishable.

## 📝 Code Desciption

`distribution.py`: read data in pandas and use seaborn to visualize the cluster distribution. 

`doc2vec_memberA.py`:

`w2v_bin_memberB.py`: 
