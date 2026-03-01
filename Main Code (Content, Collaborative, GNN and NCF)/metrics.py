import numpy as np
import torch

def precision_at_k(recommended_list, actual_list, k):
    """Calculate Precision@K"""
    if k == 0:
        return 0.0
    recommended_k = recommended_list[:k]
    hits = len(set(recommended_k).intersection(set(actual_list)))
    return hits / k

def recall_at_k(recommended_list, actual_list, k):
    """Calculate Recall@K"""
    if len(actual_list) == 0:
        return 0.0
    recommended_k = recommended_list[:k]
    hits = len(set(recommended_k).intersection(set(actual_list)))
    return hits / len(actual_list)

def ndcg_at_k(recommended_list, actual_list, k):
    """Calculate NDCG@K"""
    recommended_k = recommended_list[:k]
    
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in actual_list:
            dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed
            
    # Ideal DCG calculation for binary relevance (1 if true, 0 if false)
    idcg = 0.0
    for i in range(min(k, len(actual_list))):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0:
        return 0.0
    return dcg / idcg

def evaluate_ranking(model, test_loader, k=10, device='cpu'):
    """
    Evaluates a model producing rankings using Precision, Recall, and NDCG.
    Expected test_loader returns (users, true_items) indicating positive interactions.
    """
    precisions, recalls, ndcgs = [], [], []
    
    # Needs implementation specific to model structure (predicting unseen items)
    # This acts as a template for evaluating any given user's recommendations vs ground truth.
    return {
        f"Precision@{k}": np.mean(precisions) if precisions else 0.0,
        f"Recall@{k}": np.mean(recalls) if recalls else 0.0,
        f"NDCG@{k}": np.mean(ndcgs) if ndcgs else 0.0
    }

if __name__ == "__main__":
    # Test metrics
    rec = [10, 20, 30, 40, 50]
    act = [30, 10, 80]
    print(f"Precision@5: {precision_at_k(rec, act, 5)}")
    print(f"Recall@5: {recall_at_k(rec, act, 5)}")
    print(f"NDCG@5: {ndcg_at_k(rec, act, 5)}")
