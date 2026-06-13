import torch
from torch.utils.data import Dataset
import pandas as pd

class StabilityDataset(Dataset):
    def __init__(self, csv_path, embeddings, smiles_column="smiles", target_column="shelf_life_days"):
        self.data = pd.read_csv(csv_path)
        self.embeddings = embeddings
        self.smiles_column = smiles_column
        self.target_column = target_column
        # Only keep rows where embeddings exist
        self.valid_indices = [
            i for i, s in enumerate(self.data[self.smiles_column])
            if s in embeddings
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.data.iloc[real_idx]
        smiles = row[self.smiles_column]
        y = torch.tensor(row[self.target_column], dtype=torch.float)
        x = self.embeddings[smiles]  # 256-D tensor
        return x, y
import torch.nn as nn

class StabilityPredictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()
