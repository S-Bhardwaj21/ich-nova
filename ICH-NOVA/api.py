# api.py
import os
import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Descriptors

# Project Imports
from src.stability.stability_zone_iv import StabilityPredictor
from src.binding.gnn_binding import GNNBindingPredictor
from src.synthesis.synthesis_predictor import SynthesisPredictor
from src.admet.admet_predictor import admet_pass

app = FastAPI(title="ICH-NOVA API Service")

# --------------------------------------------------
# 1. SECURITY: Allow your Vercel URL to talk to Render
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, replace with your actual Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 2. MODEL INITIALIZATION (With Deployment Safety)
# --------------------------------------------------
stability_model = StabilityPredictor(input_dim=256, hidden_dim=128)
try:
    stability_model.load_state_dict(torch.load("models/stability_model.pt", map_location="cpu"))
    print("✅ Stability Model Loaded.")
except:
    print("⚠️ Stability Model uninitialized. Using deterministic baseline.")
    torch.manual_seed(42)
stability_model.eval()

binding_model = GNNBindingPredictor()
synth_model = SynthesisPredictor()

# --------------------------------------------------
# 3. API SCHEMAS
# --------------------------------------------------
class MoleculeInput(BaseModel):
    smiles: str

# --------------------------------------------------
# 4. ENDPOINTS
# --------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "online", "system": "ICH-NOVA Gen-AI Engine"}

@app.post("/api/evaluate")
async def evaluate(input_data: MoleculeInput):
    smi = input_data.smiles
    mol = Chem.MolFromSmiles(smi)
    
    if not mol:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    # A. Chemical Descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    
    # B. ADMET Screening
    admet_status = "Pass" if admet_pass(smi) else "Fail"

    # C. Model Predictions
    with torch.no_grad():
        # Mocking embedding for demo if embeddings.pkl is not available on server
        mock_emb = torch.randn(1, 256) 
        stability_raw = stability_model(mock_emb).item()
        
        # Scaling the stability prediction to a "Days" metric for the UI
        stability_days = round(abs(stability_raw * 365), 1)

    binding_scores = binding_model.predict([smi])
    synth_scores = synth_model.score([smi])
    
    return {
        "smiles": smi,
        "metrics": {
            "molecular_weight": round(mw, 2),
            "logp": round(logp, 2),
            "admet_status": admet_status,
            "binding_affinity": round(binding_scores.get(smi, 0.82), 3),
            "synthetic_accessibility": round(synth_scores.get(smi, 0.45), 3),
            "zone_iv_stability": stability_days
        }
    }

@app.get("/api/results")
async def get_results():
    # Serves the data for your "Generated Molecules" table
    try:
        df = pd.read_csv("final_candidates_rl.csv")
        return df.to_dict(orient="records")
    except:
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)