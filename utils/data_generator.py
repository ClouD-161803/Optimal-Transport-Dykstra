import numpy as np

def generate_crescent_data(num_particles: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a 2D crescent-shaped point cloud by applying a nonlinear shear 
    to a standard bivariate normal distribution.

    Args:
        num_particles (int): The number of particles (M) to generate.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - zeta (M x 2 array): The unperturbed standard normal particles.
            - z (M x 2 array): The transformed crescent-shaped particles.
    """
    if seed is not None:
        np.random.seed(seed)

    zeta = np.random.randn(num_particles, 2)

    z = np.zeros_like(zeta)
    
    # nonlinear shearing
    z[:, 0] = zeta[:, 0]
    z[:, 1] = zeta[:, 1] + (zeta[:, 0] ** 2)

    return zeta, z