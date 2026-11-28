import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

strengths = np.linspace(0.0, 2.0, 81)
steps = 5000

results = []

for g in strengths:
    state = np.array([0.1, 0.1])

    W = np.array([
        [0.0, g],
        [g, 0.0]
    ])

    for _ in range(steps):
        state = np.tanh(W @ state)

    results.append({
        "g": g,
        "final_activity": state[0],
        "final_activity_neuron2": state[1],
        "final_mean_activity": np.mean(state)
    })

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "connectivity_strength.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

plt.figure(figsize=(8, 5))
plt.plot(
    strengths,
    [r["final_activity"] for r in results],
    marker="o",
    markersize=3
)

plt.axvline(1.0, linestyle="--", label="Theoretical stability boundary (g=1)")
plt.xlabel("Connection strength (g)")
plt.ylabel("Final activity")
plt.title("Effect of Connectivity Strength on Network Activity")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = output_dir / "connectivity_strength.png"
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"CSV saved to: {csv_path}")
print(f"Plot saved to: {plot_path}")
