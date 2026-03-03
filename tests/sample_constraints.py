import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import hermite_polynomial

def build_dykstra_constraints(z1: np.ndarray, degree: int, epsilon: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs the polyhedral constraint matrix and offset vector for Dykstra's 
    algorithm. Enforces the strict monotonicity condition: 
    partial_1 S^1(z_1) >= epsilon.

    Because the Dykstra solver in `new_dykstra.py` expects the constraints 
    in the form N * w <= c, we formulate A * w >= epsilon as -A * w <= -epsilon.

    Args:
        z1 (np.ndarray): 1D array of the first coordinates of the particle cloud (M particles).
        degree (int): Maximum degree D of the Hermite polynomial map.
        epsilon (float, optional): The strict positivity threshold to prevent folding. Defaults to 1e-4.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - N (M x (D+1) array): The negative constraint matrix (-A).
            - c (M array): The negative threshold vector (-epsilon).
    """
    M = len(z1)
    
    # partial_1 S^1 = sum_{j=1}^D w_j * j * He_{j-1}(z_1)
    He = hermite_polynomial(z1, degree - 1)
    
    A = np.zeros((M, degree + 1))
    
    # A[:, 0] corresponds to w_0, whose derivative is 0, so it remains 0.
    
    for j in range(1, degree + 1):
        A[:, j] = j * He[:, j - 1]
    
    return -A, -epsilon * np.ones(M)

if __name__ == "__main__":
    z1_samples = np.array([0.500, 0.501, 0.502])
    D = 3
    eps = 1e-4
    
    N_matrix, c_vector = build_dykstra_constraints(z1_samples, degree=D, epsilon=eps)
    
    print(f"\nEvaluating for D={D}, epsilon={eps} at z1 = {z1_samples}")
    print("\nConstraint matrix -A:")
    print(np.round(N_matrix, 4))
    print("\nOffset vector:")
    print(c_vector)