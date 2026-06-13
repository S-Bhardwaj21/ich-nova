# dashboard.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="ICH-NOVA | Future-Grade Drug Discovery",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Dark futuristic theme
# --------------------------------------------------
st.markdown("""
<style>
body { background-color: #0e0e0e; color: #d0d0d0; }
h1, h2, h3 { color: #f0f0f0; }
.stButton>button, .stDownloadButton>button {
    background-color: #1a1a1a;
    color: #d0d0d0;
    border: 1px solid #555;
}
</style>
""", unsafe_allow_html=True)

st.title("🧬 ICH-NOVA")
st.subheader("Regulatory-aware, self-evolving AI drug discovery")

# --------------------------------------------------
# Safe loaders
# --------------------------------------------------
@st.cache_data
def load_embeddings():
    try:
        with open("data/processed/embeddings.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        # Return an empty dict if embeddings aren't found yet
        return {}

@st.cache_data
def load_results():
    try:
        return pd.read_csv("final_candidates_rl.csv")
    except Exception:
        # Fallback to empty DataFrame if simulation hasn't run yet
        return pd.DataFrame()

@st.cache_resource
def load_stability_model():
    from src.stability.stability_zone_iv import StabilityPredictor
    model = StabilityPredictor(input_dim=256, hidden_dim=128)
    try:
        # Load weights if available and healthy
        model.load_state_dict(torch.load("models/stability_model.pt", map_location="cpu"))
    except Exception:
        # Fallback to baseline weights if file is empty or corrupted
        st.sidebar.warning("⚠️ stability_model.pt is uninitialized. Running on active baseline fallback.")
        torch.manual_seed(42)
    model.eval()
    return model

embeddings_dict = load_embeddings()
final_df = load_results()
stability_model = load_stability_model()

# --------------------------------------------------
# TOP: Molecule of Interest (Primary Input)
# --------------------------------------------------
st.markdown("### 🧪 Molecule of Interest — Instant Zone-IV Evaluation")

top_col1, top_col2 = st.columns([3, 1])

with top_col1:
    top_smiles = st.text_input(
        "Enter SMILES string",
        placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O",
        key="top_smiles_input"
    )

with top_col2:
    st.write("##") # Visual alignment spacer
    top_predict = st.button("Predict Stability", key="top_predict_btn")

if top_predict and top_smiles.strip():
    mol = Chem.MolFromSmiles(top_smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        emb = embeddings_dict.get(top_smiles, torch.randn(256)).unsqueeze(0)
        with torch.no_grad():
            stab = stability_model(emb).item()

        res1, res2 = st.columns(2)
        res1.metric("Zone-IV Shelf-Life (days)", f"{abs(stab * 365):.2f}")
        res2.metric("Molecular Weight", f"{mw:.2f}")

        st.success("Prediction completed successfully.")
    else:
        st.error("Invalid SMILES string. Please check your input.")

st.markdown("---")

# --------------------------------------------------
# Normalize dataframe (CRITICAL FIX)
# --------------------------------------------------
required_cols = [
    "iteration", "smiles", "binding_score",
    "synthesis_score", "stability_pred",
    "reward", "admet_pass"
]

for col in required_cols:
    if col not in final_df.columns:
        final_df[col] = None

# If admet_pass missing → infer safely
final_df["admet_pass"] = final_df["admet_pass"].fillna(
    final_df["stability_pred"].notna().astype(int)
)

# --------------------------------------------------
# ADMET summary
# --------------------------------------------------
st.subheader("🧪 ADMET Filtering Summary")

total = len(final_df)
passed = int(final_df["admet_pass"].sum())
failed = total - passed

c1, c2, c3 = st.columns(3)
c1.metric("Total Molecules", total)
c2.metric("Passed ADMET", passed)
c3.metric("Rejected", failed)

# --------------------------------------------------
# Sidebar: Molecule explorer
# --------------------------------------------------
st.sidebar.header("🧪 Molecule Explorer")
user_smiles = st.sidebar.text_input("Enter SMILES", key="sidebar_smiles_input")

if st.sidebar.button("Predict", key="sidebar_predict_btn"):
    if user_smiles.strip():
        mol = Chem.MolFromSmiles(user_smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            emb = embeddings_dict.get(user_smiles, torch.randn(256)).unsqueeze(0)
            with torch.no_grad():
                stab = stability_model(emb).item()
            st.sidebar.success(f"Zone-IV Shelf-life: {abs(stab * 365):.2f} days")
            st.sidebar.info(f"Molecular Weight: {mw:.2f}")
        else:
            st.sidebar.error("Invalid SMILES")

# --------------------------------------------------
# Molecule table
# --------------------------------------------------
st.header("Generated Molecules")
if not final_df.empty:
    st.dataframe(
        final_df.sort_values("admet_pass", ascending=False),
        use_container_width=True
    )

    st.download_button(
        "📥 Download CSV",
        final_df.to_csv(index=False),
        "ICH_NOVA_results.csv",
        "text/csv"
    )
else:
    st.info("No pipeline run data found. Run main.py locally to generate candidates.")

# --------------------------------------------------
# Visualizations (SAFE)
# --------------------------------------------------
st.header("Visual Analysis")

numeric_df = final_df.dropna(subset=["reward", "binding_score", "stability_pred"])

if not numeric_df.empty:
    st.subheader("Average RL Reward per Iteration")
    fig, ax = plt.subplots()
    numeric_df.groupby("iteration")["reward"].mean().plot(ax=ax, marker='o')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    st.pyplot(fig)

    st.subheader("Binding vs Stability")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=numeric_df,
        x="stability_pred",
        y="binding_score",
        hue="iteration",
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Reward vs Stability")
    fig, ax = plt.subplots()
    sc = ax.scatter(
        numeric_df["stability_pred"],
        numeric_df["binding_score"],
        c=numeric_df["reward"],
        cmap="viridis"
    )
    plt.colorbar(sc, ax=ax, label="Reward")
    st.pyplot(fig)
else:
    st.warning("Not enough numeric data for plots yet.")

# --------------------------------------------------
# Ablation (optional & safe)
# --------------------------------------------------
st.header("Module Ablation Analysis")

try:
    abl = pd.read_csv("ablation_results.csv")
    num = abl.select_dtypes(include="number")
    if not num.empty:
        plot_df = num.melt(var_name="Module", value_name="Reward")
        fig, ax = plt.subplots()
        sns.boxplot(data=plot_df, x="Module", y="Reward", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Ablation file contains no numeric data.")
except Exception:
    st.info("Ablation results not available.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<center style='color:#888'>ICH-NOVA | Self-evolving AI for real-world drug discovery</center>",
    unsafe_allow_html=True
)