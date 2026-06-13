import torch
import pickle

# Example SMILES strings
smiles_list = ["CC(=O)OC1=CC=CC=C1C(=O)O", "C1CCCCC1", "CCN(CC)CC"]

# Create 256-D random embeddings
embeddings_dict = {smi: torch.randn(256) for smi in smiles_list}

# Save to embeddings.pkl
with open("data/processed/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings_dict, f)

print("Saved dummy embeddings to data/processed/embeddings.pkl")
