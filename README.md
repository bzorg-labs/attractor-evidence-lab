# Attractor Evidence Lab

A computational neuroscience project investigating attractor dynamics, perturbation resistance, and recovery in simplified recurrent neural networks.

## Research Question

Can a recurrent neural network recover its original stable state after external perturbations, and how does recovery behavior change as perturbation noise increases?

## Model

The network uses a simplified two-neuron recurrent architecture:

\[
x_{t+1} = \tanh(Wx_t)
\]

where:

\[
W =
\begin{bmatrix}
0 & 1.5 \\
1.5 & 0
\end{bmatrix}
\]

Gaussian noise is introduced during perturbation experiments.

## Experiments

### 1. Attractor Recovery

The network is perturbed with Gaussian noise. We measure its distance from the original stable state during recovery.

### 2. Decay Control

A control model follows a simple decay rule:

```python
state = 0.8 * state
```

This tests whether apparent recovery can be explained by activity decay alone.

### 3. Noise Recovery Analysis

We evaluate recovery across multiple noise levels and random seeds.

- 21 noise levels
- 100 random seeds per noise level
- 2,100 total trials
- Recovery rate and 95% Wilson confidence intervals
- Median recovery time among successful trials
- Mean final error

## Results

The multi-seed experiment shows recovery rates varying across noise levels.

In the higher tested noise range (0.414–0.420), recovery rates were approximately 29–31%.

The results suggest that recovery becomes less reliable under stronger perturbations, although confidence intervals overlap and the experiment does not establish a precise noise threshold.

Median recovery time among successful trials was approximately 4 steps in this range.

## Visualizations

- `results/control_comparison.png`
- `results/recovery_analysis.png`
- `results/multi_seed_recovery_rate.png`

## Data

- `results/multi_seed_raw.csv`: Trial-level results.
- `results/multi_seed_summary.csv`: Aggregated recovery statistics and confidence intervals.
- `results/recovery_results.csv`: Recovery experiment results.
- `results/basin_results.csv`: Basin-of-attraction experiment results.

## Limitations

- This is a highly simplified two-neuron computational model.
- It is not a biologically realistic neural network.
- The results do not demonstrate biological memory or prove attractor mechanisms in the brain.
- Recovery depends on the selected model, noise protocol, and recovery criteria.
- Further model validation, systematic controls, and comparisons with neural data are needed.

## Technologies

- Python
- NumPy
- Matplotlib
- Pandas
- Pytest

## Project Status

An exploratory computational neuroscience project focused on attractor recovery and robustness under perturbation.