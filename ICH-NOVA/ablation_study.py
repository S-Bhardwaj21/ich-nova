import torch
import pickle
import pandas as pd
from tqdm import trange

from src.stability.stability_zone_iv import StabilityPredictor
from src.generator.de_novo_generator import GraphDiffusionGenerator
from src.binding.gnn_binding import GNNBindingPredictor
from src.synthesis.synthesis_predictor import SynthesisPredictor

# ----------------------------
# Load embeddings
# ----------------------------
with open("data/processed/embeddings.pkl", "rb") as f:
    embeddings_dict = pickle.load(f)

# ----------------------------
# Load models
# ----------------------------
stability_model = StabilityPredictor()
stability_model.load_state_dict(torch.load("models/stability_model.pt"))
stability_model.eval()

generator = GraphDiffusionGenerator()
binding_model = GNNBindingPredictor()
synth_model = SynthesisPredictor()

# ----------------------------
# Experiment function
# ----------------------------
def run_experiment(use_stability=True, iterations=5, batch_size=20):
    records = []

    for it in trange(iterations, desc=f"Stability={'ON' if use_stability else 'OFF'}"):
        smiles_list = generator.generate("data/protein_target.fasta", batch_size)

        for smi in smiles_list:
            emb = embeddings_dict.get(smi)
            if emb is None:
                continue

            emb = emb.unsqueeze(0)
            binding = binding_model.predict([smi])[smi]
            synthesis = synth_model.score([smi])[smi]

            if use_stability:
                with torch.no_grad():
                    stability = stability_model(emb).item()
            else:
                stability = None

            records.append({
                "iteration": it + 1,
                "smiles": smi,
                "binding": binding,
                "stability": stability,
                "synthesis": synthesis,
                "mode": "stability_aware" if use_stability else "stability_blind"
            })

    return pd.DataFrame(records)

# ----------------------------
# Run both experiments
# ----------------------------
df_stable = run_experiment(use_stability=True)
df_blind = run_experiment(use_stability=False)

final_df = pd.concat([df_stable, df_blind], ignore_index=True)
final_df.to_csv("ablation_results.csv", index=False)

print("Saved ablation_results.csv")
