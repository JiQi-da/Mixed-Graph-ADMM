# ADMM demo

ADMM of the optimization without unrolling.

## Graph weights

- Connections: $k$ nearest neighbors, same as [Unrolling-GSP-STForecast](https://github.com/JiQi-da/Unrolling-GSP-STForecast)
- Graph weights: use a simple construction with $w_{ij}=\exp(-d_{ij}/\sigma)$, where $d_{ij}$ is the shortest path from Dijkstra Algorithm, and $j$ is one of $i$'s $k$NNs.

## ADMM algorithm
In this part we test about 3 methods:

1. Optimize $\mathbf{x}$ directly, and then optimize $\phi$.
2. Introduce auxiliary variables $\mathbf{z}_u,\mathbf{z}_d$, optimize $\mathbf{x}$ in the inner loop together with $\mathbf{z}_u,\mathbf{z}_d$, then optimize $\phi$ in the outer loop.
3. Introduce auxiliary variables $\mathbf{z}_u,\mathbf{z}_d$, optimize $\mathbf{x}, \mathbf{z}_u,\mathbf{z}_d$ and $\phi$ in the same loop.

## Mission
To find what leads to `NaN` in computing 