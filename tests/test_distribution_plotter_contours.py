import os
import sys

import matplotlib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plotter import DistributionPlotter


def test_density_grid_shape_and_finite_values() -> None:
    rng = np.random.default_rng(123)
    samples = rng.normal(size=(500, 2))
    plotter = DistributionPlotter(output_dir=os.path.join("results", "tmp_test_density"))

    bins = 48
    xx, yy, zz = plotter._estimate_density_grid(
        samples=samples,
        bins=bins,
        smooth_sigma=1.0,
    )

    assert xx.shape == (bins, bins)
    assert yy.shape == (bins, bins)
    assert zz.shape == (bins, bins)
    assert np.all(np.isfinite(zz))
    assert float(np.max(zz)) >= 0.0


def test_distribution_panel_with_contours_adds_line_collections() -> None:
    rng = np.random.default_rng(321)
    samples = rng.normal(size=(600, 2))
    plotter = DistributionPlotter(output_dir=os.path.join("results", "tmp_test_panel"))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plotter._draw_distribution_panel(
        ax=ax,
        samples=samples,
        title="Test panel",
        xlabel="$x_1$",
        ylabel="$x_2$",
        style="reference",
        draw_contours=True,
        contour_bins=64,
        contour_levels=6,
        contour_smoothing_sigma=1.2,
    )

    assert len(ax.collections) > 1
    plt.close(fig)


def test_plot_distributions_saves_with_new_defaults() -> None:
    rng = np.random.default_rng(42)
    reference_samples = rng.normal(size=(1000, 2))
    sheared_samples = reference_samples.copy()
    sheared_samples[:, 1] = sheared_samples[:, 1] + 0.75 * reference_samples[:, 0] ** 2

    output_dir = os.path.join("results", "tmp_test_plot_distributions")
    filename = "contour_default_plot.png"
    plotter = DistributionPlotter(output_dir=output_dir)
    fig = plotter.plot_distributions(
        reference_samples=reference_samples,
        sheared_samples=sheared_samples,
        seed=42,
        m=1000,
        filename=filename,
        show=False,
    )

    assert fig is not None
    assert os.path.exists(os.path.join(output_dir, filename))
    assert set(plotter.styles.keys()) >= {"reference", "sheared", "mapped"}
    plt.close(fig)
