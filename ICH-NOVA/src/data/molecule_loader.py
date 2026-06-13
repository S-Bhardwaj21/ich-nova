import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from src.data.smiles_parser import smiles_to_mol
from src.representations.molecular_graph import mol_to_graph


class MoleculeDataset:
    """
    Dataset loader for large-scale molecular datasets.
    """

    def __init__(self, csv_path, smiles_column="smiles"):
        self.data = pd.read_csv(csv_path)
        self.smiles_column = smiles_column

    def process(self, limit=None):
        graphs = []

        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            if limit and idx >= limit:
                break

            smiles = row[self.smiles_column]
            mol = smiles_to_mol(smiles)

            if mol is None:
                continue

            graph = mol_to_graph(mol)
            graphs.append(graph)

        return graphs
