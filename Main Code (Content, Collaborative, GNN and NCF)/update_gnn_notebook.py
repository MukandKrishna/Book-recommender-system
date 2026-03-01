import json
import os

NOTEBOOK_PATH = "e:/Repos/Book-recommender-system/Main/GNN.ipynb"

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find where nx.Graph() starts or ratings20 starts to cut the old graph code
cut_idx = len(nb['cells'])
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = "".join(cell.get('source', []))
        if 'G = nx.Graph()' in src or 'ratings20' in src:
            cut_idx = i
            break

nb['cells'] = nb['cells'][:cut_idx]

def create_markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split('\n')]
    }

def create_code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split('\n')]
    }

cells_to_add = [
    create_markdown("### 2. Index Mapping\nMap `User-ID` and `ISBN` to integer indices for graph construction:"),
    create_code("""# Index Mapping
user_to_index = {user: idx for idx, user in enumerate(ratings['User-ID'].unique())}
book_to_index = {book: idx for idx, book in enumerate(ratings['ISBN'].unique())}
ratings['user_index'] = ratings['User-ID'].map(user_to_index)
ratings['book_index'] = ratings['ISBN'].map(book_to_index)

print(f"Total Users: {len(user_to_index)}, Total Books: {len(book_to_index)}")"""),
    
    create_markdown("### 3. Graph Construction\nBuild edge list and node features:"),
    create_code("""import torch

edge_index = torch.tensor([ratings['user_index'].values, ratings['book_index'].values], dtype=torch.long)
edge_attr = torch.tensor(ratings['Book-Rating'].values, dtype=torch.float)

num_users = len(user_to_index)
num_books = len(book_to_index)
user_features = torch.eye(num_users)
book_features = torch.eye(num_books)
features = torch.cat([user_features, book_features])

# Note: edge_index needs an offset for the books since features are concatenated 
# User nodes are 0 to num_users-1, Book nodes are num_users to num_users+num_books-1
offset_edge_index = edge_index.clone()
offset_edge_index[1, :] += num_users"""),

    create_markdown("### 4. PyTorch Geometric Data Object"),
    create_code("""from torch_geometric.data import Data

data = Data(x=features, edge_index=offset_edge_index, edge_attr=edge_attr)
print(data)"""),

    create_markdown("### 5. Train/Test Split"),
    create_code("""import numpy as np
from sklearn.model_selection import train_test_split

train_mask, test_mask = train_test_split(np.arange(data.num_edges), test_size=0.2, random_state=42)
train_mask = torch.tensor(train_mask)
test_mask = torch.tensor(test_mask)

print(f"Train edges: {len(train_mask)}, Test edges: {len(test_mask)}")"""),

    create_markdown("### 6. Model Definition\nUsing GCNConv layers:"),
    create_code("""import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNRecSys(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64):
        super(GNNRecSys, self).__init__()
        # Initial Embeddings
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # GCN layers
        self.conv1 = GCNConv(embedding_dim, 64)
        self.conv2 = GCNConv(64, 32)
        
        # Output layers (Predicting rating)
        self.linear1 = nn.Linear(32 * 2, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, edge_index, user_indices, book_indices):
        # We start directly from embedding layer instead of `x` (one-hot matrices are very sparse and memory intensive)
        x = self.embedding.weight 
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Extract the node representations and concatenate them to predict edge attributes (ratings)
        user_h = x[user_indices]
        book_h = x[book_indices]
        
        out = torch.cat([user_h, book_h], dim=1)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        
        return out.squeeze()"""),

    create_markdown("### 7. Training and Evaluation"),
    create_code("""import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_NODES = num_users + num_books
model = GNNRecSys(NUM_NODES, embedding_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

data = data.to(device)
train_edge_index = data.edge_index[:, train_mask]
train_edge_attr = data.edge_attr[train_mask]
test_edge_index = data.edge_index[:, test_mask]
test_edge_attr = data.edge_attr[test_mask]

epochs = 50
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass on train edges
    user_indices = train_edge_index[0]
    book_indices = train_edge_index[1]
    
    preds = model(data.edge_index, user_indices, book_indices)
    loss = criterion(preds, train_edge_attr)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            t_user_indices = test_edge_index[0]
            t_book_indices = test_edge_index[1]
            t_preds = model(data.edge_index, t_user_indices, t_book_indices)
            
            val_loss = criterion(t_preds, test_edge_attr)
            val_rmse = torch.sqrt(val_loss)
            
        print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val RMSE: {val_rmse.item():.4f}")
"""),

    create_markdown("### 8. Recommendation Function\nGenerate Top-N Recommendations"),
    create_code("""def recommend_books(user_id, top_n=5):
    model.eval()
    
    if user_id not in user_to_index:
        return []
    
    u_idx = user_to_index[user_id]
    
    # Get all books the user has NOT rated yet
    user_rated_books = set(ratings[ratings['User-ID'] == user_id]['ISBN'].values)
    all_books = ratings['ISBN'].unique()
    unrated_books = [b for b in all_books if b not in user_rated_books]
    
    # Only keep unrated books that are in the book_to_index mapping
    unrated_books_indices = [book_to_index[b] + num_users for b in unrated_books if b in book_to_index]
    
    # Prepare batch
    u_tensor = torch.tensor([u_idx] * len(unrated_books_indices), dtype=torch.long).to(device)
    b_tensor = torch.tensor(unrated_books_indices, dtype=torch.long).to(device)
    
    with torch.no_grad():
        preds = model(data.edge_index, u_tensor, b_tensor)
        
    # Get top N max predictions
    _, top_indices = torch.topk(preds, top_n)
    
    recommended_isbns = [unrated_books[i.item()] for i in top_indices]
    
    # Retrieve Titles
    rec_books = books[books['ISBN'].isin(recommended_isbns)]['Book-Title'].unique()
    
    return rec_books

# Example Usage
sample_user = ratings['User-ID'].iloc[0]
print(f"Top recommendations for user {sample_user}:")
for title in recommend_books(sample_user, top_n=5):
    print(f"- {title}")""")
]

nb['cells'].extend(cells_to_add)

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with PyG GNN Pipeline!")
