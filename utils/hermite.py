import numpy as np

def hermite_polynomial(x: np.ndarray, max_degree: int) -> np.ndarray:
    """
    Evaluates probabilist's Hermite polynomials He_j(x) up to a maximum degree
    using the standard recurrence relation:
        He_0(x) = 1
        He_1(x) = x
        He_{j+1}(x) = x * He_j(x) - j * He_{j-1}(x)

    Args:
        x (np.ndarray): 1D array of input points (e.g., z_1 coordinates of particles).
        max_degree (int): The maximum polynomial degree to evaluate.

    Returns:
        np.ndarray: Matrix of shape (len(x), max_degree + 1) where the j-th 
                    column contains He_j(x) evaluated at all points.
    """
    M = len(x)
    He = np.zeros((M, max_degree + 1))
    
    if max_degree >= 0:
        He[:, 0] = 1.0
    if max_degree >= 1:
        He[:, 1] = x
        
    for j in range(1, max_degree):
        He[:, j + 1] = x * He[:, j] - j * He[:, j - 1]
        
    return He