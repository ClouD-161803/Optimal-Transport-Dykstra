"""Data source interfaces for synthetic and fixed datasets."""

from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DataBatch:
    """Canonical experiment batch shared across data pipelines."""

    reference_samples: np.ndarray
    target_samples: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """Common interface for all experiment data providers."""

    @abstractmethod
    def load(
        self,
        num_particles: int,
        num_dimensions: int,
        seed: int | None = None,
    ) -> DataBatch:
        """Load data in the experiment-native format."""


class SyntheticDataSource(DataSource):
    """Adapter for existing synthetic generators in ``utils.data_generator``."""

    def __init__(self, generator: Any) -> None:
        self.generator = generator

    def load(
        self,
        num_particles: int,
        num_dimensions: int,
        seed: int | None = None,
    ) -> DataBatch:
        reference_samples, target_samples = self.generator.generate(
            num_particles=num_particles,
            num_dimensions=num_dimensions,
            seed=seed,
        )
        return DataBatch(
            reference_samples=np.asarray(reference_samples, dtype=float),
            target_samples=np.asarray(target_samples, dtype=float),
            metadata={
                "source": "synthetic",
                "generator_class": type(self.generator).__name__,
            },
        )


class DatasetDataSource(DataSource):
    """Placeholder for fixed-dataset parsing pipeline."""

    def __init__(
        self,
        prior_csv_path: str | None = None,
        posterior_csv_path: str | None = None,
        data_root: str | None = None,
    ) -> None:
        # Default to the currently tracked Lorenz dataset files.
        resolved_root = data_root or os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "Lorenz 1963 and Feedback Particle Filter",
            "prediction_flow_data",
        )
        self.prior_csv_path = prior_csv_path or os.path.join(resolved_root, "prior.csv")
        self.posterior_csv_path = posterior_csv_path or os.path.join(
            resolved_root, "posterior.csv"
        )

    @staticmethod
    def _read_particle_csv(path: str) -> tuple[list[str], list[str], np.ndarray]:
        """Read CSV where rows are state dims and columns are particles."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset CSV not found: {path}")

        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))

        if len(rows) < 2:
            raise ValueError(f"Dataset CSV has no data rows: {path}")

        header = rows[0]
        if len(header) < 2:
            raise ValueError(
                "Dataset CSV must contain particle columns after the leading index column."
            )
        particle_labels = [label.strip() for label in header[1:]]

        state_labels: list[str] = []
        values: list[list[float]] = []
        for row in rows[1:]:
            if len(row) != len(header):
                raise ValueError(
                    "Dataset CSV row length mismatch. "
                    f"Expected {len(header)} columns, got {len(row)}."
                )
            state_labels.append(row[0].strip())
            values.append([float(cell) for cell in row[1:]])

        matrix = np.asarray(values, dtype=float).T  # (num_particles, num_dimensions)
        return particle_labels, state_labels, matrix

    def load(
        self,
        num_particles: int,
        num_dimensions: int,
        seed: int | None = None,
    ) -> DataBatch:
        if num_particles <= 0:
            raise ValueError("num_particles must be positive.")
        if num_dimensions <= 0:
            raise ValueError("num_dimensions must be positive.")

        prior_particles, prior_states, prior_matrix = self._read_particle_csv(
            self.prior_csv_path
        )
        posterior_particles, posterior_states, posterior_matrix = self._read_particle_csv(
            self.posterior_csv_path
        )

        if prior_particles != posterior_particles:
            raise ValueError(
                "prior.csv and posterior.csv do not share the same particle ordering."
            )
        if prior_states != posterior_states:
            raise ValueError(
                "prior.csv and posterior.csv do not share the same state-dimension labels."
            )
        if prior_matrix.shape != posterior_matrix.shape:
            raise ValueError(
                "prior.csv and posterior.csv shape mismatch: "
                f"{prior_matrix.shape} vs {posterior_matrix.shape}."
            )

        available_particles, available_dimensions = prior_matrix.shape
        if num_dimensions > available_dimensions:
            raise ValueError(
                f"Requested num_dimensions={num_dimensions}, but dataset only has "
                f"{available_dimensions} dimensions."
            )
        if num_particles > available_particles:
            raise ValueError(
                f"Requested num_particles={num_particles}, but dataset only has "
                f"{available_particles} particles."
            )

        prior_trimmed = prior_matrix[:, :num_dimensions]
        posterior_trimmed = posterior_matrix[:, :num_dimensions]

        if num_particles < available_particles:
            rng = np.random.default_rng(seed)
            selected_idx = rng.choice(
                available_particles,
                size=num_particles,
                replace=False,
            )
            prior_trimmed = prior_trimmed[selected_idx]
            posterior_trimmed = posterior_trimmed[selected_idx]

        # Keep the same contract as synthetic generation:
        # reference_samples first, source/sheared-like samples second.
        # Here: posterior is treated as reference/target, prior as source.
        return DataBatch(
            reference_samples=np.asarray(posterior_trimmed, dtype=float),
            target_samples=np.asarray(prior_trimmed, dtype=float),
            metadata={
                "source": "dataset",
                "prior_csv_path": os.path.abspath(self.prior_csv_path),
                "posterior_csv_path": os.path.abspath(self.posterior_csv_path),
                "state_labels": tuple(prior_states[:num_dimensions]),
                "num_available_particles": available_particles,
                "num_available_dimensions": available_dimensions,
                "num_loaded_particles": num_particles,
                "num_loaded_dimensions": num_dimensions,
            },
        )
