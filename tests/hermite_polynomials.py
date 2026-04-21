import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.hermite import hermite_polynomial

def test_hermite_polynomials():
    x = np.linspace(-3, 3, 100)
    Psi = hermite_polynomial(x, DEGREE)

    plt.figure(figsize=(10, 4))
    for j in range(DEGREE + 1):
        plt.plot(x, Psi[:, j], label=r"$\Psi_{%d}(y)$" % j)
    # plt.title(f"Hermite polynomials up to degree {DEGREE}", fontsize=20)
    plt.xlabel(r"y", fontsize=16)
    plt.ylabel(r"$\Psi_j(y)$", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=13)
    # plt.tight_layout(rect=[0, 0, 1.0, 1])

if __name__ == "__main__":
    DEGREE = 5

    test_hermite_polynomials()
    plt.show()

