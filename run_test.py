
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.simulation import (
    simulate_connectivity_network,
    simulate_noise_recovery,
)

W = np.array([
    [0.0, 1.5],
    [1.5, 0.0]
])

baseline = simulate_connectivity_network(
    W, steps=500
)[-1]

noise_levels = np.round(
    np.arange(0.38, 0.422, 0.002), 3
)

seeds = list(range(100))

tolerance = 0.01
noise_steps = 100
recovery_steps = 400
stable_window = 10

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

all_results = []
summary_results = []

print("Baseline:", np.round(baseline, 5))
print("\nMULTI-SEED EXPERIMENT")
print("-" * 75)

for noise in noise_levels:

    recovery_times = []
    final_errors = []

    for seed in seeds:

        trajectory = simulate_noise_recovery(
            W,
            baseline,
            noise_std=float(noise),
            noise_steps=noise_steps,
            recovery_steps=recovery_steps,
            seed=seed
        )

        post_noise = trajectory[noise_steps:]

        errors = np.linalg.norm(
            post_noise - baseline,
            axis=1
        )

        recovered_indices = [
            i for i in range(
                len(errors) - stable_window + 1
            )
            if np.all(errors[i:] <= tolerance)
        ]
        
        recovery_time = (
            recovered_indices[0] + 1
            if recovered_indices
            else None
        )

        final_error = float(errors[-1])
        recovered = recovery_time is not None

        if recovered:
            recovery_times.append(recovery_time)

        final_errors.append(final_error)

        all_results.append({
            "noise": float(noise),
            "seed": seed,
            "recovered": recovered,
            "recovery_time": recovery_time,
            "final_error": final_error
        })

    recovery_rate = len(recovery_times) / len(seeds)

    median_time = (
        float(np.median(recovery_times))
        if recovery_times
        else None
    )

    mean_final_error = float(np.mean(final_errors))

    summary_results.append({
        "noise": float(noise),
        "recovery_rate": recovery_rate,
        "median_recovery_time": median_time,
        "mean_final_error": mean_final_error
    })

    print(
        f"Noise={noise:.3f} | "
        f"Recovery Rate={recovery_rate:.0%} | "
        f"Median Time={median_time} | "
        f"Mean Final Error={mean_final_error:.5f}"
    )


# Save individual trial results
with open(
    results_dir / "multi_seed_raw.csv",
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "noise",
            "seed",
            "recovered",
            "recovery_time",
            "final_error"
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)


# Save aggregated results
with open(
    results_dir / "multi_seed_summary.csv",
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "noise",
            "recovery_rate",
            "median_recovery_time",
            "mean_final_error"
        ]
    )

    writer.writeheader()
    writer.writerows(summary_results)


# Plot recovery rate
noise_values = [
    item["noise"] for item in summary_results
]

recovery_rates = [
    item["recovery_rate"] * 100
    for item in summary_results
]

plt.figure(figsize=(9, 5))

plt.plot(
    noise_values,
    recovery_rates,
    marker="o"
)

plt.xlabel("Noise Standard Deviation")
plt.ylabel("Recovery Rate (%)")
plt.title("Attractor Recovery Rate vs Noise Level")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_dir / "multi_seed_recovery_rate.png",
    dpi=200
)

plt.show()

print("\nFiles saved:")
print("results/multi_seed_raw.csv")
print("results/multi_seed_summary.csv")
print("results/multi_seed_recovery_rate.png")
