#!/usr/bin/env python3
"""Generate particle samples for optimal-transport TikZ visualization."""

from __future__ import annotations

import argparse
import re
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_generator import ShearFunction, DataGenerator
from utils.plotter import DistributionPlotter

N_PARTICLES = 50
SEED = 42
ALPHA = 0.7
SHIFT = -0.5
PLOT_SIZE = 3.0


class QuadraticShearFunction(ShearFunction):
    """x2 = z2 + alpha*z1^2 + shift"""

    def __init__(self, alpha: float = ALPHA, shift: float = SHIFT) -> None:
        self.alpha = float(alpha)
        self.shift = float(shift)

    def shear(self, zeta: np.ndarray) -> np.ndarray:
        return self.alpha * zeta[:, 0] ** 2 + self.shift


def fmt_num(v: float) -> str:
    """Clean up small values for TikZ."""
    if abs(v) < 5e-4:
        v = 0.0
    return f"{v:.2f}"


def tikz_node_lines(points: np.ndarray, style: str) -> str:
    return "\n".join(
        f"\\node[{style}] at ({fmt_num(x)}, {fmt_num(y)}) {{}};"
        for x, y in points
    )


def build_left_block(z_cloud: np.ndarray) -> str:
    return tikz_node_lines(z_cloud, "mapped_particle")


def build_right_block(x_cloud: np.ndarray) -> str:
    return tikz_node_lines(x_cloud, "particle")


def replace_block(tex: str, begin_marker: str, end_marker: str, replacement: str) -> str:
    pattern = re.compile(
        rf"({re.escape(begin_marker)}\n)(.*?)(\n{re.escape(end_marker)})",
        flags=re.DOTALL,
    )
    if not pattern.search(tex):
        raise ValueError(f"Could not find markers: {begin_marker} ... {end_marker}")
    return pattern.sub(rf"\1{replacement}\3", tex, count=1)


def patch_template(template_path: Path, out_path: Path, left_block: str, right_block: str) -> None:
    tex = template_path.read_text(encoding="utf-8")
    tex = replace_block(tex, "% __LEFT_PARTICLES_BEGIN__", "% __LEFT_PARTICLES_END__", left_block)
    tex = replace_block(tex, "% __RIGHT_PARTICLES_BEGIN__", "% __RIGHT_PARTICLES_END__", right_block)
    out_path.write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=None, help="Optional TikZ/LaTeX file to patch.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-cloud", type=int, default=N_PARTICLES)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--shift", type=float, default=SHIFT)
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent / "results" / "data_generation" / "latex_visual"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_tex = results_dir / "particles.tex"

    shear_func = QuadraticShearFunction(alpha=args.alpha, shift=args.shift)
    generator = DataGenerator(shear_function=shear_func)
    z_cloud, x_cloud = generator.generate(
        num_particles=args.n_cloud,
        num_dimensions=2,
        seed=args.seed,
    )

    plotter = DistributionPlotter(output_dir=str(results_dir))
    plotter.plot_distributions(
        reference_samples=z_cloud,
        sheared_samples=x_cloud,
        seed=args.seed,
        m=args.n_cloud,
        shear_label="Quadratic",
        xlim=(-PLOT_SIZE, PLOT_SIZE),
        ylim=(-PLOT_SIZE, PLOT_SIZE),
        show=False,
    )

    left_block = build_left_block(z_cloud)
    right_block = build_right_block(x_cloud)

    if args.template is not None:
        patch_template(args.template, out_tex, left_block, right_block)
    else:
        out_text = (
            "% Auto-generated particle blocks\n\n"
            "% LEFT PANEL\n"
            f"{left_block}\n\n"
            "% RIGHT PANEL\n"
            f"{right_block}\n"
        )
        out_tex.write_text(out_text, encoding="utf-8")

    print(f"Wrote LaTeX: {out_tex}")
    print(f"Wrote plots to: {results_dir}")


if __name__ == "__main__":
    main()
