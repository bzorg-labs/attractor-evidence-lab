
import numpy as np
import csv
from src.simulation import simulate_decay_control
from pathlib import Path

from src.simulation import simulate_bistable_attractor


def test_attractor_recovery(
    perturbation_strength=0.2,
    steps=200,
    seed=42
):
    baseline = simulate_bistable_attractor(
        steps=steps,
        seed=seed
    )

    stable_state = baseline[-1].copy()

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        0,
        perturbation_strength,
        size=stable_state.shape
    )

    disturbed_state = stable_state + noise

    recovery = simulate_bistable_attractor(
        steps=steps,
        seed=seed,
        initial_state=disturbed_state
    )

    initial_distance = np.linalg.norm(
        disturbed_state - stable_state
    )

    distances = np.linalg.norm(
        recovery - stable_state,
        axis=1
    )

    return initial_distance, distances


def run_repeated_trials(
    noise_levels=(1, 2, 3, 4, 5, 10, 15, 20),
    n_trials=30,
    steps=200,
    threshold=0.01
):
    from src.visualization import plot_recovery_results

    recovery_rates = []
    recovery_times = []

    for noise in noise_levels:
        successful = 0
        trial_times = []

        for seed in range(n_trials):
            initial, distances = test_attractor_recovery(
                perturbation_strength=noise,
                steps=steps,
                seed=seed
            )

            recovered = np.where(distances < threshold)[0]

            if len(recovered) > 0:
                successful += 1
                trial_times.append(recovered[0] + 1)

        rate = successful / n_trials * 100
        mean_time = np.mean(trial_times) if trial_times else float("nan")

        recovery_rates.append(rate)
        recovery_times.append(mean_time)

        print(
            f"Noise: {noise:>2} | "
            f"Recovery: {rate:.1f}% | "
            f"Mean recovery time: {mean_time:.1f}"
        )

    Path("results").mkdir(exist_ok=True)

    with open("results/recovery_results.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Noise",
            "Recovery Rate",
            "Mean Recovery Time"
        ])

        for noise, rate, time in zip(
            noise_levels, recovery_rates, recovery_times
        ):
            writer.writerow([noise, rate, time])

    plot_recovery_results(
        noise_levels,
        recovery_rates,
        recovery_times
    )
def test_decay_control(
    perturbation_strength=0.2,
    steps=200,
    seed=42
):
    baseline = simulate_bistable_attractor(
        steps=steps,
        seed=seed
    )

    stable_state = baseline[-1].copy()

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        0,
        perturbation_strength,
        size=stable_state.shape
    )

    disturbed_state = stable_state + noise

    activity = simulate_decay_control(
        steps=steps,
        seed=seed,
        initial_state=disturbed_state
    )

    distances = np.linalg.norm(
        activity - stable_state,
        axis=1
    )

    return activity, distances

def estimate_basin_of_attraction(
    perturbation_levels=(0.1, 0.5, 1, 2, 3, 5),
    n_trials=30,
    steps=200,
    threshold=0.1,
    seed=42
):
    baseline = simulate_bistable_attractor(
        steps=steps,
        seed=seed
    )

    stable_state = baseline[-1].copy()
    rng = np.random.default_rng(seed)

    results = []

    for level in perturbation_levels:
        successful = 0

        for _ in range(n_trials):
            noise = rng.normal(
                0, level, size=stable_state.shape
            )

            initial_state = stable_state + noise

            activity = simulate_bistable_attractor(
                steps=steps,
                seed=seed,
                initial_state=initial_state
            )

            final_distance = np.linalg.norm(
                activity[-1] - stable_state
            )

            if final_distance < threshold:
                successful += 1

        recovery_rate = successful / n_trials * 100

        results.append((level, recovery_rate))

        print(
            f"Perturbation: {level} | "
            f"Basin recovery: {recovery_rate:.1f}%"
        )

    Path("results").mkdir(exist_ok=True)

    with open(
        "results/basin_results.csv",
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Perturbation", "Recovery Rate"])
        writer.writerows(results)

    return results

def test_multiple_networks(
    n_networks=30,
    perturbation=50,
    n_trials=100,
    steps=200,
    threshold=0.1
):
    network_rates = []

    for network_seed in range(n_networks):
        baseline = simulate_bistable_attractor(
            steps=steps,
            seed=network_seed
        )

        stable_state = baseline[-1].copy()
        rng = np.random.default_rng(1000)

        noises = rng.normal(
            0, perturbation,
            size=(n_trials, len(stable_state))
        )

        successes = 0

        for noise in noises:
            activity = simulate_bistable_attractor(
                steps=steps,
                seed=network_seed,
                initial_state=stable_state + noise
            )

            distance = np.linalg.norm(
                activity[-1] - stable_state
            )

            if distance < threshold:
                successes += 1

        rate = successes / n_trials * 100
        network_rates.append(rate)

        print(f"Network {network_seed + 1}: {rate:.1f}%")

    print(f"Mean recovery: {np.mean(network_rates):.1f}%")
    print(f"SD: {np.std(network_rates, ddof=1):.1f}%")

    return network_rates