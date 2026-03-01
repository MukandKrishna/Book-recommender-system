import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LightGCN as PyG_LightGCN

class MatrixFactorization(nn.Module):
    """
    Standard Matrix Factorization (SVD equivalent in deep learning).
    Learns user and item embeddings and predicts score as their dot product.
    """
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings properly
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)
        return (user_embeds * item_embeds).sum(dim=1)

class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (NCF).
    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP).
    """
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64, 32]):
        super().__init__()
        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_modules = []
        input_size = embedding_dim * 2
        for size in hidden_layers:
            mlp_modules.append(nn.Linear(input_size, size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=0.2))
            input_size = size
        self.mlp_layers = nn.Sequential(*mlp_modules)
        
        # Final prediction layer
        self.predict_layer = nn.Linear(embedding_dim + hidden_layers[-1], 1)
        
        # Initialization
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        # GMF part
        user_gmf = self.user_embedding_gmf(user_indices)
        item_gmf = self.item_embedding_gmf(item_indices)
        gmf_vector = user_gmf * item_gmf
        
        # MLP part
        user_mlp = self.user_embedding_mlp(user_indices)
        item_mlp = self.item_embedding_mlp(item_indices)
        mlp_vector = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)
        
        # Combine
        combined_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        prediction = self.predict_layer(combined_vector)
        
        return prediction.squeeze()

class LightGCNRecommender(nn.Module):
    """
    Wrapper around PyTorch Geometric's LightGCN implementation for Graph-based recommendation.
    """
    def __init__(self, num_nodes, embedding_dim=64, num_layers=3):
        super().__init__()
        self.model = PyG_LightGCN(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            num_layers=num_layers
        )
        
    def forward(self, edge_index):
        # Returns the embeddings of all nodes
        return self.model.get_embedding(edge_index)
        
    def recommend(self, edge_index, src_index, dst_index):
        """
        Returns predictions (inner product) for the given user (src) and item (dst) pairs.
        """
        out = self.model.get_embedding(edge_index)
        src_out = out[src_index]
        dst_out = out[dst_index]
        return (src_out * dst_out).sum(dim=-1)

    def calculate_loss(self, edge_index, pos_edge_label_index):
        """
        Calculates the BPR loss safely avoiding PyTorch CPU log_sigmoid bugs.
        """
        # 1. Get embeddings
        out = self.model.get_embedding(edge_index)
        
        # 2. Sample negative edges (since we only pass positive interactions)
        # Using PyG's structured negative sampling or random sampling
        from torch_geometric.utils import structured_negative_sampling
        # structured_negative_sampling returns (i, j, k) where (i,j) is positive and (i,k) is negative
        # But our edges might be bipartite, so we need to ensure the negatives are from the correct set
        # Let's do a simple random sampling for negatives for robust compatibility
        num_users = pos_edge_label_index[0].max().item() + 1
        num_nodes = out.size(0)
        num_items = num_nodes - num_users
        
        # Sample random negatives (item indices)
        neg_items = torch.randint(0, num_items, (pos_edge_label_index.size(1),), device=out.device) + num_users
        
        # 3. Calculate scores
        user_emb = out[pos_edge_label_index[0]]
        pos_item_emb = out[pos_edge_label_index[1]]
        neg_item_emb = out[neg_items]
        
        pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=-1)
        
        # 4. Calculate BPR Loss strictly in float32
        # log_sigmoid works on floats, not longs
        loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
        
        # Add slight L2 regularization to embeddings (standard for LightGCN)
        reg_loss = (user_emb.norm(p=2).pow(2) + 
                    pos_item_emb.norm(p=2).pow(2) + 
                    neg_item_emb.norm(p=2).pow(2)) / (2 * len(user_emb))
        
        return loss + (1e-4 * reg_loss)

if __name__ == "__main__":
    num_users = 100
    num_items = 200
    num_nodes = num_users + num_items
    
    # Test MF
    mf = MatrixFactorization(num_users, num_items)
    print("MF Output shape:", mf(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])).shape)
    
    # Test NCF
    ncf = NeuralCollaborativeFiltering(num_users, num_items)
    print("NCF Output shape:", ncf(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])).shape)
    
    # Test LightGCN
    lgcn = LightGCNRecommender(num_nodes)
    edge_index = torch.tensor([[0, 1, 2], [101, 102, 103]], dtype=torch.long)
    print("LightGCN Emb shape:", lgcn(edge_index).shape)
