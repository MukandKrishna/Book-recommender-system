import torch
from torch import optim
from models import MatrixFactorization, NeuralCollaborativeFiltering, LightGCNRecommender
from dataset import BookRecommendationDataset, get_ncf_dataloaders
import torch.nn.functional as F

def train_ncf(model, train_loader, epochs=10, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # BCEWithLogitsLoss since our NCF outputs raw logits and labels are 1s/0s
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if len(batch) == 3:
                users, items, labels = [x.to(device) for x in batch]
            else:
                users, items = [x.to(device) for x in batch]
                labels = torch.ones_like(users, dtype=torch.float)
            
            optimizer.zero_grad()
            predictions = model(users, items)
            
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1:03d}/{epochs} | NCF Train Loss: {total_loss/len(train_loader):.4f}")

def train_mf(model, train_loader, epochs=10, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # BCEWithLogitsLoss for implicit feedback 1s/0s ranking approximation
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if len(batch) == 3:
                users, items, labels = [x.to(device) for x in batch]
            else:
                users, items = [x.to(device) for x in batch]
                labels = torch.ones_like(users, dtype=torch.float)
            
            optimizer.zero_grad()
            predictions = model(users, items)
            
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1:03d}/{epochs} | MF Train Loss: {total_loss/len(train_loader):.4f}")

def train_lightgcn(model, data, train_mask, epochs=100, lr=0.01, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Use only the training edges
    train_edge_index = data.edge_index[:, train_mask]
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # PyG LightGCN samples negatives internally for BPR loss using its built-in method
        loss = model.calculate_loss(train_edge_index.to(device), train_edge_index.to(device))
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | LightGCN BPR Loss: {loss.item():.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = BookRecommendationDataset()
    
    # --- Train NCF ---
    print("\n--- Training NCF ---")
    ncf_model = NeuralCollaborativeFiltering(dataset.num_users, dataset.num_books)
    train_loader, test_loader = get_ncf_dataloaders(dataset.ratings, batch_size=2048)
    
    # Reduced epochs for quick test
    train_ncf(ncf_model, train_loader, epochs=3, device=device)
    
    # --- Train LightGCN ---
    print("\n--- Training LightGCN ---")
    gnn_data, train_mask, test_mask = dataset.get_gnn_data()
    lgcn_model = LightGCNRecommender(dataset.num_users + dataset.num_books)
    
    # Reduced epochs for quick test
    train_lightgcn(lgcn_model, gnn_data, train_mask, epochs=20, device=device)
