
import numpy as np

from src.simulation import (
    simulate_bistable_attractor,
    simulate_decay_control,
    simulate_connectivity_network,
    simulate_with_noise,
    simulate_noise_recovery,
)


W = np.array([
    [0.0, 1.5],
    [1.5, 0.0]
])

INITIAL_STATE = np.array([0.2, 0.4])


def test_attractor_output_shape():
    activity = simulate_bistable_attractor(
        n_neurons=20,
        steps=50
    )
    assert activity.shape == (50, 20)


def test_attractor_is_reproducible():
    first = simulate_bistable_attractor(
        n_neurons=20, steps=50, seed=42
    )
    second = simulate_bistable_attractor(
        n_neurons=20, steps=50, seed=42
    )
    assert np.allclose(first, second)


def test_decay_activity_decreases():
    activity = simulate_decay_control(
        n_neurons=20, steps=50, seed=42
    )
    norms = np.linalg.norm(activity, axis=1)
    assert np.all(np.diff(norms) < 0)


def test_simulate_with_noise_output_shape():
    activity = simulate_with_noise(
        W, steps=50, initial_state=INITIAL_STATE
    )
    assert activity.shape == (50, 2)


def test_simulate_with_noise_is_reproducible():
    first = simulate_with_noise(
        W, steps=50, initial_state=INITIAL_STATE, seed=42
    )
    second = simulate_with_noise(
        W, steps=50, initial_state=INITIAL_STATE, seed=42
    )
    assert np.allclose(first, second)


def test_noise_recovery_output_shape():
    activity = simulate_noise_recovery(
        W,
        INITIAL_STATE,
        noise_steps=30,
        recovery_steps=40
    )
    assert activity.shape == (70, 2)


def test_zero_noise_matches_deterministic_network():
    recovery = simulate_noise_recovery(
        W,
        INITIAL_STATE,
        noise_std=0,
        noise_steps=10,
        recovery_steps=20
    )

    deterministic = simulate_connectivity_network(
        W,
        steps=30,
        initial_state=INITIAL_STATE
    )

    assert np.allclose(recovery, deterministic)
