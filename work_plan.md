in Jupyter Notebook:
- visualize a neighborhood of each node, see the signal trend (correlated or not?)

in `ADMM.py`:
- [Done] dual residuals $\Vert x^{\tau+1} - x^\tau\Vert$ and delta x for each time step
- the differentials, share the paper with Gene

- [50%] convergence speed of undirected version
    - [Done] uniformed operation: $\mathbf{L}^n \mathbf{x}$. Only 1 regularized term in final version. In Line graph/kNN graph 2 versions.
    - Opitimization fomulation for this ablation study.

in `utils.py`:
- [Done] change directed graph into line graphs: $\mathbf{L}^d_r=\begin{bmatrix}
0 &\\
-1 & 1 &\\
 & -1 & 1\\
 & & \ddots &\ddots \\
 & & & -1 & 1\end{bmatrix}$ because of random-walk regularization
- change edge weights into only 0/1 to see the connections
- condition number of $\mathbf{L}^u, \mathbf{L}^d$. Especially $\mathbf{L}^u$.
- [Done] Try interpolation problem.

---
Done:
- Visualizing regularization terms DGTV, DGLR, GLR
- adding dual residual
- interpolation problem
- line graph