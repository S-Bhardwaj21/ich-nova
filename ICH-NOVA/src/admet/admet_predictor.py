from rdkit import Chem
from rdkit.Chem import Descriptors

def admet_pass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    # Simple Lipinski-style filter
    if mw < 500 and logp < 5 and hbd <= 5 and hba <= 10:
        return 1
    return 0
