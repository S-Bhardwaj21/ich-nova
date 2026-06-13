from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_mol(smiles: str):
    """
    Convert SMILES string to RDKit Mol object with sanitization.

    Args:
        smiles (str): SMILES representation of molecule

    Returns:
        rdkit.Chem.Mol or None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def mol_to_fingerprint(mol, radius=2, n_bits=2048):
    """
    Generate Morgan fingerprint for a molecule.

    Used later for:
    - Stability QSAR
    - Applicability Domain estimation
    """
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits
    )
