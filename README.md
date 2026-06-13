# 🧬 ICH-NOVA

### *Climate-Aware Computational Drug Discovery & Autonomous Molecular Architecture*

---

![License](https://img.shields.io/badge/license-MIT-09090b?style=flat-square&labelColor=1a1a1a)
![Python](https://img.shields.io/badge/python-3.9%2B-09090b?style=flat-square&logo=python&labelColor=1a1a1a)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-09090b?style=flat-square&logo=pytorch&labelColor=1a1a1a)
![Next.js](https://img.shields.io/badge/Next.js-14-09090b?style=flat-square&logo=nextdotjs&labelColor=1a1a1a)

ICH-NOVA is an elite, industrial-grade computational drug discovery pipeline that bridges deep generative biochemistry with global regulatory intelligence. By coupling **Target-Conditioned *De Novo* Drug Design** using **3D Geometric Diffusion Models** with an advanced **ICH Zone IV Stability Prediction Engine**, ICH-NOVA engineers candidates that are optimized not just for binding affinity, but for real-world global manufacturing and stability under extreme climatic conditions.

The platform is engineered as a decoupled, production-ready SaaS application featuring a premium, obsidian-and-gold luxury dashboard for high-throughput screening and analytical visualization.

---

## 🏛️ Core System Architecture

ICH-NOVA operates as a closed-loop, self-evolving system utilizing multi-objective reinforcement learning across four core pillars:
[ Target Protein FASTA ]
│
▼
┌─────────────────────────────────┐
│ 1. 3D Geometric Diffusion Gen   │ ──► Generates pockets-fit 3D ligands
└─────────────────────────────────┘
│
▼
┌─────────────────────────────────┐
│ 2. Deep GNN Binding Predictor   │ ──► Screens for strict target affinity
└─────────────────────────────────┘
│
▼
┌─────────────────────────────────┐
│ 3. Automated ADMET & SA Filters │ ──► Rejects toxic & unsynthesizable fragments
└─────────────────────────────────┘
│
▼
┌─────────────────────────────────┐     Calculates high-temperature/humidity
│ 4. ICH Zone IV Stability Module │ ──► degradation risks (Hydrolysis/Oxidation)
└─────────────────────────────────┘
│
▼
[ Multi-Objective RL Reward Loop ] ──► Continuously optimizes the Generator
### Key Technical Highlights
* **Target-Conditioned Geometric Diffusion:** Leverages 3D equivariant neural networks to dynamically generate novel molecular structures directly inside the target protein's binding pocket from structural FASTA sequences.
* **ICH Zone IV Climate Predictive Modeling:** A specialized Graph Neural Network (GNN) and Transformer-based regression network that forecasts chemical degradation, shelf-life, and structural resilience under International Council for Harmonisation (ICH) Zone IV environmental parameters (hot/humid equatorial climates).
* **Multi-Objective Reinforcement Learning:** Coordinates standard binding kinetics ($K_d$/$K_i$) with a synthetic accessibility penalty and stability coefficients to dynamically evolve the chemical space over multiple training iterations.

---

## 💻 Tech Stack & Decoupled Infrastructure

The engineering architecture is strictly decoupled to ensure high-performance computation independent of user interface overhead:

* **The Brain (Backend Engine):** Built with **Python, FastAPI, and PyTorch**. It acts as the core machine learning inference microservice handling graph neural network evaluations, molecular feature maps, and chemical properties computing via **RDKit**. Hosted natively on **Render**.
* **The Face (Interactive Dashboard):** A premium, luxury-tier user interface designed with a tailored obsidian-black aesthetic. Built using **Next.js, Tailwind CSS, Lucide Icons,** and **Shadcn/ui primitives**. Hosted on **Vercel**.

---

## 📂 Repository Structure

```text
├── data/                       # Dataset pipelines (Raw FASTA & Processed Embeddings)
├── models/                     # Serialized PyTorch model weights (.pt checkpoint layers)
├── src/                        # Core algorithmic submodules
│   ├── generator/              # 3D Geometric Graph Diffusion networks
│   ├── binding/                # GNN-based pocket binding predictors
│   ├── stability/              # ICH Zone IV deep regression modeling
│   ├── synthesis/              # Synthetic Accessibility scoring engines
│   └── admet/                  # Toxicity and pharmacokinetics filtering pipelines
├── api.py                      # FastAPI orchestration microservice layer
├── dashboard.py                # Standalone data visualization interface
├── requirements.txt            # Python ecosystem dependency configurations
└── README.md                   # System documentation
```
🚀 Live Production Deployment
The ecosystem is split into production environments across Vercel and Render:

Backend API Microservice: https://ich-nova-6w9p.onrender.com

Frontend Analytics Platform: https://ich-nova-frontend.vercel.app/

Local Development Quickstart
To spin up the computing engine microservice locally:

Bash
# Clone the repository
git clone [https://github.com/yourusername/ICH-NOVA.git](https://github.com/S-Bhardwaj21/ICH-NOVA.git)
cd ICH-NOVA

# Install specific machine learning and chemistry dependencies
pip install -r requirements.txt

# Launch the local Uvicorn ASGI server
uvicorn api:app --host 0.0.0.0 --port 8000

