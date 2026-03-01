import torch
import numpy as np
from collections import defaultdict
from models import NeuralCollaborativeFiltering, LightGCNRecommender
from metrics import precision_at_k, recall_at_k, ndcg_at_k
from dataset import BookRecommendationDataset
from tqdm import tqdm

def evaluate_ranking_model(model, dataset, test_mask, k=10, device='cpu'):
    model.eval()
    
    gnn_data, train_mask, _ = dataset.get_gnn_data(test_size=0.2)
    
    train_edge_index = gnn_data.edge_index[:, train_mask]
    test_edge_index = gnn_data.edge_index[:, test_mask]
    
    # Group train and test items by user
    train_user_items = defaultdict(set)
    test_user_items = defaultdict(set)
    
    print("Preparing evaluation data...")
    for i in range(train_edge_index.size(1)):
        u = train_edge_index[0, i].item()
        v = train_edge_index[1, i].item() - dataset.num_users
        train_user_items[u].add(v)
        
    for i in range(test_edge_index.size(1)):
        u = test_edge_index[0, i].item()
        v = test_edge_index[1, i].item() - dataset.num_users
        test_user_items[u].add(v)
        
    precisions, recalls, ndcgs = [], [], []
    all_books_tensor = torch.arange(dataset.num_books, dtype=torch.long, device=device)
    
    # Let's evaluate a sample of users to save time during this quick test
    test_users = list(test_user_items.keys())[:100]
    print(f"Evaluating on {len(test_users)} test users...")
    
    with torch.no_grad():
        if isinstance(model, LightGCNRecommender):
            out = model.model.get_embedding(train_edge_index.to(device))
            
        for u in test_users: # use tqdm(test_users) in real script
            u_tensor = torch.tensor([u], dtype=torch.long, device=device)
            u_tensor_expanded = u_tensor.expand(dataset.num_books)
            
            if isinstance(model, NeuralCollaborativeFiltering):
                preds = model(u_tensor_expanded, all_books_tensor)
            elif isinstance(model, LightGCNRecommender):
                preds = out[u_tensor_expanded] * out[all_books_tensor + dataset.num_users]
                preds = preds.sum(dim=-1)
            else: # MatrixFactorization
                preds = model(u_tensor_expanded, all_books_tensor)
                
            preds = preds.cpu().numpy()
            
            # Mask out train items
            train_items = list(train_user_items[u])
            if train_items:
                preds[train_items] = -np.inf
            
            top_k_indices = np.argsort(preds)[::-1][:k]
            actual_items = list(test_user_items[u])
            
            precisions.append(precision_at_k(top_k_indices, actual_items, k))
            recalls.append(recall_at_k(top_k_indices, actual_items, k))
            ndcgs.append(ndcg_at_k(top_k_indices, actual_items, k))
        
    return {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        f"NDCG@{k}": np.mean(ndcgs)
    }
