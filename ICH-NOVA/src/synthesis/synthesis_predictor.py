# synthesis/synthesis_predictor.py
import torch

class SynthesisPredictor:
    def __init__(self, model_path=None):
        # Load reaction feasibility model if available
        self.model_path = model_path
        if model_path:
            self.model = torch.load(model_path)
        else:
            self.model = None

    def score(self, smiles_list):
        """
        Predict synthesis feasibility for each molecule.
        Returns a dict {smiles: score}.
        """
        scores = {}
        for smi in smiles_list:
            # Dummy score: higher is easier to synthesize
            scores[smi] = torch.rand(1).item() * 5
        return scores
