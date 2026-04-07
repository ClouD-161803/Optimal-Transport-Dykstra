"""Data source interfaces for synthetic and fixed datasets."""

from __future__ import annotations

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

    def load(
        self,
        num_particles: int,
        num_dimensions: int,
        seed: int | None = None,
    ) -> DataBatch:
        raise NotImplementedError(
            "DatasetDataSource is scaffolding only for now. "
            "Dataset parsing is intentionally not implemented yet."
        )

