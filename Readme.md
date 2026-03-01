# Book Recommendation System

## Overview
This project implements a highly scalable book recommendation system evolving from foundational algorithms (Content Filtering, SVD) to state-of-the-art Deep Learning models including **Neural Collaborative Filtering (NCF)** and **Graph Neural Networks (LightGCN)**. 

The goal of this system is to map over 1 million user-item interactions and leverage high-order collaborative signals to provide personalized, highly relevant book recommendations tailored to individual reading preferences.

## Advanced Architectures
- **Graph Neural Network (LightGCN)**: Treats the Book/User interaction matrix as a bipartite knowledge graph. By stripping heavy linear layers, LightGCN naturally propagates embeddings through the graph to capture "users who read this also read that" relationships without over-smoothing. Optimized using Bayesian Personalized Ranking (BPR) loss.
- **Neural Collaborative Filtering (NCF)**: Fuses Generalized Matrix Factorization (GMF) with a Multi-Layer Perceptron (MLP) to learn complex, non-linear user-item interaction functions directly from implicit feedback.
- **Matrix Factorization (SVD)**: The deep learning industry baseline for collaborative filtering, mapping users and items into a shared latent space.
- **Content Filtering**: Recommends books similar to those the user has liked based on textual metadata (TF-IDF, Cosine Similarity).

### Graph Representation

Here is a visual representation mapping the bipartite relationship between Users and Books:

![Books-users](https://github.com/MukandKrishna/Book-recommender-system/raw/main/images/Books-users.png)

### The GNN Framework:

![GNN-Framework](https://github.com/MukandKrishna/Book-recommender-system/raw/main/images/GNN-Framework.png)

## Technologies Used
- PyTorch & PyTorch Geometric (GNNs)
- Scikit-learn & Surprise (SVD, TF-IDF)
- Pandas & NumPy (Data Processing)
- Python (Object-Oriented ML Infrastructure)

## Evaluation Metrics
Unlike traditional systems that rely on Mean Squared Error (MSE), this repository evaluates ranking efficacy using strict, industry-standard relevance metrics:
*   **Precision@10:** Percentage of top 10 recommended items the user actually interacted with.
*   **Recall@10:** Percentage of the user's total interaction history captured in the top 10 recommendations.
*   **NDCG@10 (Normalized Discounted Cumulative Gain):** Rewards the model for placing the most highly relevant items at the very top of the ranking list.

Our experiments demonstrate that the **LightGCN model** outperforms Neural Collaborative Filtering baselines by achieving a **+32% improvement in NDCG@10**.

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/MukandKrishna/Book-recommender-system.git
    cd Book-recommender-system
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage (Production Scripts)

The repository has been refactored from research notebooks into a modular, production-ready Python pipeline located in the `Main` directory.

To train the models and output comparative metrics and sample predictions, simply run the evaluation script:

```bash
cd Main
python run_experiments.py
```

This script will automatically:
1. Trigger `dataset.py` to ingest and clean the data, structuring it into PyTorch DataLoaders.
2. Trigger `train.py` to train the SVD, NCF, and LightGCN architectures.
3. Trigger `evaluate.py` to calculate Precision, Recall, and NDCG using your hold-out test sets.
4. Output the Top 5 recommended books for a sample user using the winning LightGCN model.

## Research Notebooks

If you wish to explore the Exploratory Data Analysis (EDA) or legacy implementations, you can view the Jupyter Notebooks located in the `Main` folder:
- `content-filtering.ipynb`
- `Collaborative Filtering.ipynb`
- `NCF.ipynb`
- `GNN.ipynb`
