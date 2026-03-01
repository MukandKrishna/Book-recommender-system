import torch
import pandas as pd
from tabulate import tabulate
from dataset import BookRecommendationDataset, get_ncf_dataloaders
from models import MatrixFactorization, NeuralCollaborativeFiltering, LightGCNRecommender
from train import train_mf, train_ncf, train_lightgcn
from evaluate import evaluate_ranking_model
import warnings

warnings.filterwarnings('ignore')

def run_all_experiments():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running Recommendation Experiments on {device} ---")
    
    # 1. Load Data
    print("\n[1/4] Loading Dataset...")
    dataset = BookRecommendationDataset()
    train_loader, test_loader = get_ncf_dataloaders(dataset.ratings, batch_size=2048)
    gnn_data, train_mask, test_mask = dataset.get_gnn_data(test_size=0.2)
    
    results = {}
    
    # 2. Matrix Factorization (SVD Baseline)
    print("\n[2/4] Training Matrix Factorization (SVD Baseline)...")
    mf_model = MatrixFactorization(dataset.num_users, dataset.num_books)
    train_mf(mf_model, train_loader, epochs=5, device=device)
    
    print("Evaluating Matrix Factorization...")
    mf_metrics = evaluate_ranking_model(mf_model, dataset, test_mask, k=10, device=device)
    results['SVD / MF'] = mf_metrics
    
    # 3. Neural Collaborative Filtering
    print("\n[3/4] Training Neural Collaborative Filtering (NCF)...")
    ncf_model = NeuralCollaborativeFiltering(dataset.num_users, dataset.num_books)
    train_ncf(ncf_model, train_loader, epochs=5, device=device)
    
    print("Evaluating Neural Collaborative Filtering...")
    ncf_metrics = evaluate_ranking_model(ncf_model, dataset, test_mask, k=10, device=device)
    results['NCF'] = ncf_metrics
    
    # 4. LightGCN
    print("\n[4/4] Training LightGCN...")
    lgcn_model = LightGCNRecommender(dataset.num_users + dataset.num_books)
    train_lightgcn(lgcn_model, gnn_data, train_mask, epochs=50, device=device)
    
    print("Evaluating LightGCN...")
    lgcn_metrics = evaluate_ranking_model(lgcn_model, dataset, test_mask, k=10, device=device)
    results['LightGCN'] = lgcn_metrics
    
    # Print Comparison Table
    print("\n" + "="*50)
    print("🏆 FINAL MODEL COMPARISON 🏆")
    print("="*50)
    
    table_data = []
    for model_name, metrics in results.items():
        table_data.append([
            model_name, 
            f"{metrics['Precision@10']:.4f}", 
            f"{metrics['Recall@10']:.4f}", 
            f"{metrics['NDCG@10']:.4f}"
        ])
        
    print(tabulate(table_data, headers=["Model", "Precision@10", "Recall@10", "NDCG@10"], tablefmt="grid"))
    
    # Recommendations for a sample user
    print("\n" + "="*50)
    print("📚 SAMPLE RECOMMENDATIONS (Using LightGCN) 📚")
    print("="*50)
    
    sample_user_id = dataset.ratings['User-ID'].iloc[0] # Pick the first user
    sample_user_idx = dataset.user_to_index[sample_user_id]
    
    lgcn_model.eval()
    with torch.no_grad():
        out = lgcn_model.model.get_embedding(gnn_data.edge_index.to(device))
        u_emb = out[sample_user_idx]
        all_books_emb = out[dataset.num_users : dataset.num_users + dataset.num_books]
        
        scores = (u_emb * all_books_emb).sum(dim=-1).cpu().numpy()
        
        # Remove already read books
        read_books_idx = dataset.ratings[dataset.ratings['User-ID'] == sample_user_id]['book_index'].values
        scores[read_books_idx] = -float('inf')
        
        top_5_idx = scores.argsort()[::-1][:5]
        
    print(f"Top 5 Recommended Books for User-ID: {sample_user_id}")
    for idx in top_5_idx:
        # Reverse mapping: find ISBN from index
        isbn = list(dataset.book_to_index.keys())[list(dataset.book_to_index.values()).index(idx)]
        title_matches = dataset.books[dataset.books['ISBN'] == isbn]['Book-Title'].values
        title = title_matches[0] if len(title_matches) > 0 else "Unknown Title"
        print(f" - {title} (ISBN: {isbn}) | Score: {scores[idx]:.4f}")

    print("\n" + "="*50)
    print("🤔 WHICH MODEL IS BEST & WHY?")
    print("="*50)
    print("LightGCN typically performs the best for this dataset and recommendation paradigm.")
    print("Why?")
    print("1. High-Order Connectivity: Matrix Factorization and NCF only model explicit, 1-hop interactions")
    print("   between a user and an item. LightGCN propagates embeddings through the bipartite graph, capturing")
    print("   signals like 'users who read this book also read that book'.")
    print("2. Efficiency: By removing non-linear activations and weight matrices in the GCN layers, LightGCN")
    print("   avoids over-smoothing and learns better embeddings purely focused on neighborhood aggregation.")
    print("3. BPR Loss: LightGCN natively optimizes for Bayesian Personalized Ranking, directly boosting Recall and NDCG.")

if __name__ == "__main__":
    run_all_experiments()
