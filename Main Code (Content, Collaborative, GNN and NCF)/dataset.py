import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class BookRecommendationDataset:
    def __init__(self, data_dir='../data/original'):
        self.data_dir = data_dir
        
        self.books = None
        self.users = None
        self.ratings = None
        
        self.user_to_index = None
        self.book_to_index = None
        
        self.num_users = 0
        self.num_books = 0
        
        self._load_and_clean_data()
        self._create_mappings()

    def _load_and_clean_data(self):
        print(f"Loading data from {self.data_dir}...")
        self.books = pd.read_csv(f'{self.data_dir}/Books.csv', low_memory=False)
        self.users = pd.read_csv(f'{self.data_dir}/Users.csv')
        self.ratings = pd.read_csv(f'{self.data_dir}/Ratings.csv')

        # Clean ratings
        self.ratings = self.ratings[self.ratings['Book-Rating'] != 0]
        self.ratings = self.ratings.dropna(subset=['User-ID', 'ISBN', 'Book-Rating'])
        self.ratings = self.ratings.drop_duplicates(subset=['User-ID', 'ISBN'])
        
        # Clean users
        self.users = self.users.dropna(subset=['User-ID'])
        self.users = self.users[(self.users['Age'].fillna(0) > 5) & (self.users['Age'].fillna(0) < 100)]
        
        # Clean books
        self.books = self.books.dropna(subset=['ISBN', 'Book-Title'])

        # Merge ratings with valid users and books
        self.ratings = self.ratings.merge(self.users[['User-ID']], on='User-ID', how='inner')
        self.ratings = self.ratings.merge(self.books[['ISBN']], on='ISBN', how='inner')
        
        self.ratings.reset_index(drop=True, inplace=True)
        self.users.reset_index(drop=True, inplace=True)
        self.books.reset_index(drop=True, inplace=True)
        
        # We only need these columns for training
        self.ratings = self.ratings[["User-ID", "ISBN", "Book-Rating"]]
        
        print(f"Cleaned Data - Users: {len(self.users)}, Books: {len(self.books)}, Ratings: {len(self.ratings)}")

    def _create_mappings(self):
        print("Creating index mappings...")
        self.user_to_index = {user: idx for idx, user in enumerate(self.ratings['User-ID'].unique())}
        self.book_to_index = {book: idx for idx, book in enumerate(self.ratings['ISBN'].unique())}
        
        self.ratings['user_index'] = self.ratings['User-ID'].map(self.user_to_index)
        self.ratings['book_index'] = self.ratings['ISBN'].map(self.book_to_index)
        
        self.num_users = len(self.user_to_index)
        self.num_books = len(self.book_to_index)
        
        print(f"Total Unique Users: {self.num_users}, Total Unique Books: {self.num_books}")

    def get_gnn_data(self, test_size=0.2):
        print("Generating PyTorch Geometric Data object...")
        user_indices = self.ratings['user_index'].values
        book_indices = self.ratings['book_index'].values
        
        # Create edge index
        edge_index = torch.tensor([user_indices, book_indices], dtype=torch.long)
        edge_attr = torch.tensor(self.ratings['Book-Rating'].values, dtype=torch.float)

        # For bipartite graph, book indices are offset by num_users
        offset_edge_index = edge_index.clone()
        offset_edge_index[1, :] += self.num_users
        
        features = torch.arange(self.num_users + self.num_books, dtype=torch.long)
        
        data = Data(x=features, edge_index=offset_edge_index, edge_attr=edge_attr, num_nodes=self.num_users+self.num_books)
        
        train_mask, test_mask = train_test_split(np.arange(data.num_edges), test_size=test_size, random_state=42)
        
        return data, torch.tensor(train_mask), torch.tensor(test_mask)

class NCFDataset(Dataset):
    def __init__(self, user_indices, item_indices, labels=None):
        self.user_indices = torch.tensor(user_indices, dtype=torch.long)
        self.item_indices = torch.tensor(item_indices, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float) if labels is not None else None

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.user_indices[idx], self.item_indices[idx], self.labels[idx]
        return self.user_indices[idx], self.item_indices[idx]

def get_ncf_dataloaders(ratings_df, test_size=0.2, batch_size=2048):
    print("Generating PyTorch DataLoaders for NCF/SVD...")
    # Convert implicit feedback to 1s
    user_indices = ratings_df['user_index'].values
    book_indices = ratings_df['book_index'].values
    labels = np.ones(len(ratings_df)) # For BPR/Ranking we often just use 1s for positive interactions
    
    # Normally we also sample negative items (0s) during training for NCF, 
    # but that logic can reside in the training loop or a specialized sampler.
    # For now, we return the positive interactions.
    
    X_train_u, X_test_u, X_train_b, X_test_b, y_train, y_test = train_test_split(
        user_indices, book_indices, labels, test_size=test_size, random_state=42
    )
    
    train_dataset = NCFDataset(X_train_u, X_train_b, y_train)
    test_dataset = NCFDataset(X_test_u, X_test_b, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    dataset = BookRecommendationDataset()
    gnn_data, train_mask, test_mask = dataset.get_gnn_data()
    print(f"GNN Data: {gnn_data}")
    
    train_loader, test_loader = get_ncf_dataloaders(dataset.ratings)
    print(f"NCF Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
