import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import hermite_polynomial

def test_hermite_polynomials():
    x = np.linspace(-3, 3, 100)
    He = hermite_polynomial(x, DEGREE)

    plt.figure(figsize=(10, 6))
    for j in range(DEGREE + 1):
        plt.plot(x, He[:, j], label=r"$He_{%d}(x)$" % j)
    plt.title(f"Hermite polynomials up to degree {DEGREE}")
    plt.xlabel("x")
    plt.ylabel(r"$He_j(x)$")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

if __name__ == "__main__":
    DEGREE = 5

    test_hermite_polynomials()
    plt.show()

