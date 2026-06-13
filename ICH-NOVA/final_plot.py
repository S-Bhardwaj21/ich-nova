import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ablation_results.csv")

stable = df[df["mode"] == "stability_aware"]["stability"].dropna()
blind = df[df["mode"] == "stability_blind"]["binding"]

plt.figure(figsize=(7,5))
plt.boxplot([stable, blind], labels=["Stability-Aware RL", "Stability-Blind RL"])
plt.ylabel("Predicted Shelf-Life / Binding Proxy")
plt.title("Impact of Stability Awareness on Candidate Quality")
plt.grid(True)
plt.show()
