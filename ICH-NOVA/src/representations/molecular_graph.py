from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


ATOM_LIST = [
    "H", "C", "N", "O", "F",
    "P", "S", "Cl", "Br", "I"
]


def atom_features(atom):
    """
    Encode atom features as a numeric vector.
    Regulatory-safe, interpretable features.
    """
    return np.array([
        ATOM_LIST.index(atom.GetSymbol()) if atom.GetSymbol() in ATOM_LIST else -1,
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetHybridization().real,
        atom.GetIsAromatic()
    ], dtype=np.float32)


def bond_features(bond):
    """
    Encode bond features.
    """
    return np.array([
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.IsInRing()
    ], dtype=np.float32)


def mol_to_graph(mol):
    """
    Convert RDKit Mol to graph representation.

    Returns:
        node_features: (N_atoms, F_atom)
        edge_index: (2, N_edges)
        edge_features: (N_edges, F_bond)
    """
    node_features = []
    edge_index = []
    edge_features = []

    for atom in mol.GetAtoms():
        node_features.append(atom_features(atom))

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Undirected graph: add both directions
        edge_index.append([i, j])
        edge_index.append([j, i])

        bf = bond_features(bond)
        edge_features.append(bf)
        edge_features.append(bf)

    return (
        np.array(node_features),
        np.array(edge_index).T,
        np.array(edge_features)
    )
