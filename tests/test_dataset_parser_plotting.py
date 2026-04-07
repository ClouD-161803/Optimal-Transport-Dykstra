import os
import sys
import types
import importlib

import matplotlib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
matplotlib.use("Agg")

from core.data import DatasetDataSource


def _load_distribution_plotter_class():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    utils_dir = os.path.join(repo_root, "utils")
    if "utils" not in sys.modules:
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [utils_dir]  # type: ignore[attr-defined]
        sys.modules["utils"] = utils_pkg
    module = importlib.import_module("utils.plotter")
    return module.DistributionPlotter


def test_dataset_parser_and_distribution_plot_2d_subsample() -> None:
    num_particles = 500
    num_dimensions = 2
    seed = 42
    plot_size = 30.0
    distribution_xlim: tuple[float, float] | None = (-plot_size, plot_size)
    distribution_ylim: tuple[float, float] | None = (-plot_size, plot_size)

    data_source = DatasetDataSource()
    batch = data_source.load(
        num_particles=num_particles,
        num_dimensions=num_dimensions,
        seed=seed,
    )

    reference_samples = batch.reference_samples
    sheared_samples = batch.target_samples

    assert reference_samples.shape == (num_particles, num_dimensions)
    assert sheared_samples.shape == (num_particles, num_dimensions)
    assert np.all(np.isfinite(reference_samples))
    assert np.all(np.isfinite(sheared_samples))
    assert batch.metadata["source"] == "dataset"

    output_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
        "data_generation",
    )
    filename = (
        f"dataset_distribution_SEED={seed}_M={num_particles}_"
        f"D={num_dimensions}_SOURCE=LorenzPriorPosterior.png"
    )

    DistributionPlotter = _load_distribution_plotter_class()
    plotter = DistributionPlotter(output_dir=output_dir)
    plotter.plot_distributions(
        reference_samples=reference_samples,
        sheared_samples=sheared_samples,
        seed=seed,
        m=num_particles,
        shear_label="LorenzPriorPosterior",
        reference_title="Posterior (Reference) Distribution",
        sheared_title="Prior (Sheared) Distribution",
        xlim=distribution_xlim,
        ylim=distribution_ylim,
        filename=filename,
        show=False,
    )

    assert os.path.exists(os.path.join(output_dir, filename))
