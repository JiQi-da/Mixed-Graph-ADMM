import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Solve linear system Ax = b using Conjugate Gradient method
    
    Parameters:
    A: matrix of coefficients (n x n)
    b: right hand side vector (n x 1)
    x0: initial guess (default: zero vector)
    tol: tolerance for convergence
    max_iter: maximum number of ßiterations
    
    Returns:
    x: solution vector
    n_iter: number of iterations performed
    """
    n = len(b)
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()
    
    r = b - A @ x  # residual
    p = r.copy()   # search direction
    
    r_norm_sq = r @ r
    
    for k in range(max_iter):
        Ap = A @ p
        alpha = r_norm_sq / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        
        r_norm_new_sq = r @ r
        beta = r_norm_new_sq / r_norm_sq
        r_norm_sq = r_norm_new_sq
        
        if np.sqrt(r_norm_sq) < tol:
            return x, k + 1
            
        p = r + beta * p
    
    return x, max_iter

# Example usage
if __name__ == "__main__":
    # Test with a simple system
    A = np.array([[4, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    
    x_solution, iterations = conjugate_gradient(A, b)
    print("Solution:", x_solution)
    print("Iterations:", iterations)
    print("Verification (A @ x):", A @ x_solution)
    print("Original b:", b)