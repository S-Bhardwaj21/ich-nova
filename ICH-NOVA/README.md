🧬 ICH-NOVA
A Closed-Loop, Stability-Aware AI System for Regulatory-Viable Drug Design
🔍 Project Motivation

Traditional AI-driven drug discovery systems focus on molecular potency and novelty, but often ignore real-world regulatory constraints such as climatic stability.
This leads to late-stage failures, especially for ICH Zone IV regions (India, ASEAN, Africa) where high temperature and humidity accelerate degradation.

ICH-NOVA addresses this gap by integrating ICH Zone IV stability intelligence directly into a self-evolving drug design loop, enabling the discovery of molecules that are not only effective but regulatorily viable.

🧠 System Overview

ICH-NOVA is a closed-loop AI system composed of six tightly integrated modules:

Protein Target
     ↓
De Novo Molecular Generator
     ↓
Protein–Ligand Binding Intelligence
     ↓
ADMET & Toxicity Filtering
     ↓
Synthesis Feasibility Scoring
     ↓
ICH Zone IV Stability Prediction
     ↓
Reinforcement Learning Feedback
     ↓
Self-Improving Molecular Generator

🧩 Core Modules
1️⃣ De Novo Molecular Generation

Generates novel molecules conditioned on a protein target

Molecules represented as graphs (not strings)

Designed for extensibility to diffusion/VAE models

2️⃣ Protein–Ligand Binding Prediction

Graph Neural Network–based binding affinity estimation

Acts as the primary efficacy signal

3️⃣ ADMET & Toxicity Intelligence

Filters unsafe or clinically risky molecules

Mimics early-stage clinical failure prevention

4️⃣ Synthesis Feasibility Scoring

Penalizes chemically implausible or costly molecules

Ensures manufacturability awareness

5️⃣ Reinforcement Learning Loop

Multi-objective reward function combining:

Binding

ADMET

Synthesis

Stability

Enables autonomous self-improvement

6️⃣ ICH Zone IV Stability Prediction (Key Innovation)

Predicts shelf-life under 30 °C / 65–75% RH

Includes confidence / applicability awareness

Integrates regulatory feasibility into discovery

📊 Outputs

Running the system produces:

final_candidates_rl.csv

Generated molecules

Binding scores

Stability predictions

RL rewards

Diagnostic plots:

Reward improvement over iterations

Binding vs stability trade-offs

Stability confidence visualizations

▶️ How to Run
python main.py

🧪 Scientific Significance

Unlike conventional student projects, ICH-NOVA:

Treats regulatory approval as a design constraint

Integrates stability after discovery, not as an afterthought

Demonstrates systems-level AI reasoning

Mirrors real pharmaceutical R&D decision pipelines

🚀 Future Extensions (Not Implemented Yet)

Replace dummy generators with trained diffusion models

Deploy as a web-based decision-support tool

Integrate real CDSCO / ICH stability datasets

👩‍💻 Author

Developed as a research-grade CSE undergraduate project focused on AI for pharmaceutical compliance and discovery.