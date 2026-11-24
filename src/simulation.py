
import numpy as np


def simulate_recurrent_network(
    n_neurons=100,
    steps=200,
    seed=42,
    initial_state=None
):
    rng = np.random.default_rng(seed)

    weights = rng.normal(
        0, 1 / np.sqrt(n_neurons),
        (n_neurons, n_neurons)
    )

    if initial_state is None:
        state = rng.normal(0, 0.1, n_neurons)
    else:
        state = np.array(initial_state, dtype=float).copy()

    activity = np.zeros((steps, n_neurons))

    for t in range(steps):
        state = np.tanh(weights @ state)
        activity[t] = state

    return activity



def simulate_stable_network(
    n_neurons=100,
    steps=200,
    initial_state=None
):
    if initial_state is None:
        state = np.random.default_rng(42).normal(
            0, 0.1, n_neurons
        )
    else:
        state = np.array(initial_state, dtype=float).copy()

    activity = np.zeros((steps, n_neurons))

    for t in range(steps):
        state = 0.8 * state + 0.2 * np.tanh(state)
        activity[t] = state

    return activity



def simulate_decay_control(
    n_neurons=100,
    steps=200,
    seed=42,
    initial_state=None
):
    if initial_state is None:
        rng = np.random.default_rng(seed)
        state = rng.normal(0, 0.1, n_neurons)
    else:
        state = np.array(initial_state, dtype=float).copy()

    activity = np.zeros((steps, n_neurons))

    for t in range(steps):
        state = 0.8 * state
        activity[t] = state

    return activity

def simulate_bistable_attractor(
    n_neurons=100,
    steps=200,
    seed=42,
    initial_state=None
):
    rng = np.random.default_rng(seed)

    weights = np.full(
        (n_neurons, n_neurons),
        1.5 / n_neurons
    )

    weights += rng.normal(
        0,
        0.05 / np.sqrt(n_neurons),
        (n_neurons, n_neurons)
    )

    if initial_state is None:
        state = rng.normal(0.1, 0.01, n_neurons)
    else:
        state = np.array(initial_state, dtype=float).copy()

    activity = np.zeros((steps, n_neurons))

    for t in range(steps):
        state = np.tanh(weights @ state)
        activity[t] = state

    return activity

def simulate_connectivity_network(
    connectivity,
    steps=200,
    initial_state=None
):
    connectivity = np.asarray(connectivity, dtype=float)

    if connectivity.ndim != 2:
        raise ValueError("Connectivity must be a 2D matrix")

    if connectivity.shape[0] != connectivity.shape[1]:
        raise ValueError("Connectivity matrix must be square")

    n_neurons = connectivity.shape[0]

    if initial_state is None:
        state = np.ones(n_neurons) * 0.1
    else:
        state = np.asarray(initial_state, dtype=float).copy()

    if state.shape != (n_neurons,):
        raise ValueError("Initial state size must match connectivity")

    activity = np.zeros((steps, n_neurons))

    for t in range(steps):
        state = np.tanh(connectivity @ state)
        activity[t] = state

    return activity


def simulate_with_noise(
    connectivity,
    steps=200,
    noise_std=0.05,
    initial_state=None,
    seed=42
):
    import numpy as np

    rng = np.random.default_rng(seed)
    connectivity = np.asarray(connectivity, dtype=float)

    if connectivity.ndim != 2:
        raise ValueError("Connectivity must be a 2D matrix")

    if connectivity.shape[0] != connectivity.shape[1]:
        raise ValueError("Connectivity matrix must be square")

    n = connectivity.shape[0]

    state = (
        np.ones(n) * 0.1
        if initial_state is None
        else np.asarray(initial_state, dtype=float).copy()
    )

    if state.shape != (n,):
        raise ValueError("Initial state size must match connectivity")

    activity = np.zeros((steps, n))

    for t in range(steps):
        noise = rng.normal(0, noise_std, n)
        state = np.tanh(connectivity @ state + noise)
        activity[t] = state

    return activity

def simulate_noise_recovery(
    connectivity,
    initial_state,
    noise_std=0.1,
    noise_steps=100,
    recovery_steps=400,
    seed=42
):
    rng = np.random.default_rng(seed)
    connectivity = np.asarray(connectivity, dtype=float)
    state = np.asarray(initial_state, dtype=float).copy()

    n = connectivity.shape[0]
    activity = np.zeros((noise_steps + recovery_steps, n))

    for t in range(noise_steps + recovery_steps):
        if t < noise_steps:
            noise = rng.normal(0, noise_std, n)
        else:
            noise = np.zeros(n)

        state = np.tanh(connectivity @ state + noise)
        activity[t] = state

    return activity
