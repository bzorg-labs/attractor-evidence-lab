## Connectivity Strength Experiment

### Research question
How does recurrent connection strength affect the equilibrium behavior of a two-neuron recurrent network?

### Model
x(t+1) = tanh(Wx(t))
W = [[0, g], [g, 0]]

### Experimental design
- 81 connection-strength values from 0 to 2.
- 5,000 iterations per parameter value.
- Initial state: [0.1, 0.1].

### Result
The experiment shows a transition in equilibrium behavior near g = 1. Below the critical value, activity decays toward zero. Above it, nonzero activity emerges. At g = 1, convergence is very slow, so finite-time residual activity must not be interpreted as a nonzero equilibrium.

### Mathematical interpretation
The linearization at the origin has eigenvalues ±g. The stability of the origin changes at g = 1. Above this critical value, nonzero symmetric equilibria satisfy a = tanh(g*a).

### Limitations
This is a simplified two-neuron deterministic model. It demonstrates a mathematical stability transition, not evidence of a transition in biological neural circuits.

### Reproducibility
The experiment generates a CSV dataset and a plot using Python and NumPy.
