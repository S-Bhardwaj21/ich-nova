import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import pandas as pd

from rdkit import Chem
from src.representations.molecular_graph import mol_to_graph
from core.gnn_encoder import MolecularGNNEncoder


class GraphDatasetProcessor:
    """
    Converts SMILES dataset into PyTorch Geometric Data objects
    and generates embeddings with MolecularGNNEncoder.
    """

    def __init__(self, csv_path, smiles_column="smiles", device="cpu"):
        self.data = pd.read_csv(csv_path)
        self.smiles_column = smiles_column
        self.device = device

    def smiles_to_data(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        node_features, edge_index, edge_attr = mol_to_graph(mol)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        node_features = torch.tensor(node_features, dtype=torch.float)

        # batch will be set later
        return Data(x=node_features, edge_index=edge_index)

    def process_dataset(self, limit=None):
        data_list = []

        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            if limit and idx >= limit:
                break
            smiles = row[self.smiles_column]
            data = self.smiles_to_data(smiles)
            if data is not None:
                data_list.append(data)

        return data_list

    def compute_embeddings(self, data_list, node_dim, hidden_dim=128, output_dim=256):
        model = MolecularGNNEncoder(node_dim=node_dim,
                                    hidden_dim=hidden_dim,
                                    output_dim=output_dim).to(self.device)
        model.eval()

        embeddings = []
        batch = Batch.from_data_list(data_list).to(self.device)

        with torch.no_grad():
            z = model(batch)
            embeddings.append(z.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings
