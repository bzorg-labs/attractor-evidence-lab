import matplotlib.pyplot as plt


def plot_recovery_results(noise_levels, recovery_rates, recovery_times):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(noise_levels, recovery_rates, marker="o")
    axes[0].set_title("Recovery Rate")
    axes[0].set_xlabel("Noise Strength")
    axes[0].set_ylabel("Recovery (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True)

    axes[1].plot(noise_levels, recovery_times, marker="o")
    axes[1].set_title("Mean Recovery Time")
    axes[1].set_xlabel("Noise Strength")
    axes[1].set_ylabel("Time Steps")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("results/recovery_analysis.png", dpi=300)
    plt.show()

def plot_control_comparison(
    attractor_distances,
    control_distances
):
    plt.figure(figsize=(8, 5))

    plt.plot(
        attractor_distances,
        label="Attractor Recovery"
    )

    plt.plot(
        control_distances,
        label="Decay Control"
    )

    plt.xlabel("Time Steps")
    plt.ylabel("Distance from Stable State")
    plt.title("Attractor vs Decay Control")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "results/control_comparison.png",
        dpi=300
    )

    plt.show()