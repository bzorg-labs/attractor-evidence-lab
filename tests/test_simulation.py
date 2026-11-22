import numpy as np

from src.simulation import (
    simulate_bistable_attractor,
    simulate_decay_control,
)


def test_attractor_output_shape():
    activity = simulate_bistable_attractor(
        n_neurons=20,
        steps=50
    )

    assert activity.shape == (50, 20)


def test_attractor_is_reproducible():
    first = simulate_bistable_attractor(
        n_neurons=20,
        steps=50,
        seed=42
    )

    second = simulate_bistable_attractor(
        n_neurons=20,
        steps=50,
        seed=42
    )

    assert np.allclose(first, second)


def test_decay_activity_decreases():
    activity = simulate_decay_control(
        n_neurons=20,
        steps=50,
        seed=42
    )

    norms = np.linalg.norm(activity, axis=1)

    assert np.all(np.diff(norms) < 0)