import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool


class MolecularGNNEncoder(nn.Module):
    """
    Research-grade molecular graph encoder.
    Outputs a fixed-length latent embedding for each molecule.
    """

    def __init__(self, node_dim, hidden_dim=128, output_dim=256):
        super().__init__()

        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.readout = global_mean_pool

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = self.readout(x, batch)
        z = self.projection(x)

        return z
