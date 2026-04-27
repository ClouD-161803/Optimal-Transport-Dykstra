# Optimal Transport Dykstra Algorithm

A comprehensive implementation of the Knothe-Rosenblatt (KR) rearrangement map with fast-forward Dykstra projections for optimal transport problems. This codebase provides experimental frameworks for benchmarking different solver backends (vanilla Dykstra, fast-forward Dykstra, and QP-based methods) across synthetic and real datasets.

## Overview

This project implements n-dimensional optimal transport computations using the Knothe-Rosenblatt (KR) rearrangement and projected gradient descent with Dykstra projections. The core algorithm decomposes the KR map into lower-dimensional marginal components, each optimized independently.

### Key Features
- **Fast-Forward Dykstra**: Accelerated projection algorithm for marginal constraints
- **Multi-solver Benchmarking**: Compare vanilla Dykstra vs. fast-forward vs. QP solvers
- **Synthetic & Real Data**: Support for both procedurally generated data and real datasets
- **Scalability Testing**: Runtime benchmarks across problem dimensions (2D to 50D)
- **Visualisation**: Distribution evolution and solver performance plots

## Main Experiments

### 1. Synthetic Data Experiment (`experiments/kr_map_experiment.py`)

**Purpose**: Benchmark KR map learning on procedurally generated synthetic distributions with known structure.

**Data Generators**:
- **BoomerangShearFunction** (default): Nonlinear shearing transformation creating a curved "boomerang" shape
- **RoughLineShearFunction**: Jagged line pattern with noise (configurable via `sigma` parameter)
- **CubicShearFunction**: Strong cubic nonlinearity (strength 0.4)
- **XShapedShearFunction**: X-shaped distribution (strength 1.2)
- **LayeredBoomerangShearFunction**: Multi-scale boomerang pattern for scaling experiments
- **GVMDataGenerator**: Generalized von Mises distribution (requires `alpha`, `beta`, `gamma`, `kappa` parameters)

**Configuration Highlights**:
```python
NUM_DIMENSIONS = 2              # Problem dimensionality
NUM_PARTICLES = 2500            # Number of samples
SEED = 1234                     # Reproducibility
MAX_OUTER_ITER = 17500          # Outer optimisation iterations
BATCH_SIZE = 700                # SGD batch size (None = full gradient)
LEARNING_RATE = 7e-2            # Initial learning rate
L1_REG = 0.0                    # L1 regularisation strength
```

**Running the Experiment**:
```bash
python experiments/kr_map_experiment.py
```

**Output**:
- Distribution comparison plots (prior, posterior, mapped prior for both solvers)
- Convergence metrics and solver runtimes
- Results organised in `results/full_experiment_benchmarks/distribution_plots/`

**Customisation**:
Modify the data generator by uncommenting one of the pre-defined generators:
```python
# Uncomment one generator:
DATA_GENERATOR = DataGenerator(shear_function=BoomerangShearFunction())
# DATA_GENERATOR = DataGenerator(shear_function=RoughLineShearFunction(sigma=0.15))
# DATA_GENERATOR = DataGenerator(shear_function=CubicShearFunction(strength=0.4))
```

---

### 2. Real Dataset Experiment (`experiments/kr_map_dataset_experiment.py`)

**Purpose**: Apply KR map learning to real-world data from filtering/prediction problems.

**Dataset**: Lorenz 1963 System with Feedback Particle Filter
- **Location**: `data/Lorenz 1963 and Feedback Particle Filter/prediction_flow_data/`
- **Files**: `prior.csv` (reference prior), `posterior.csv` (reference posterior)
- **Dimensions**: 3D system
- **Particles**: CSV files contain 500 particles (downsampled to configurable amount)

**Key Differences from Synthetic Experiment**:
- Data is loaded from CSV instead of generated procedurally
- **Whitening Preprocessing**: Both prior and posterior are whitened (zero mean, unit variance) before optimisation
- Inverse whitening is applied to results for physical-space plotting
- Panel titles customised for reference/prior/posterior terminology
- Limited optimisation iterations (1000 vs 17500) due to fixed dataset size

**Configuration**:
```python
NUM_DIMENSIONS = 3              # Must match dataset dimensionality
NUM_PARTICLES = 200             # Downselected from 500 available
MAX_OUTER_ITER = 1000           # Fewer iterations for real data
BATCH_SIZE = None               # Uses full-batch gradient
PLOT_DISTRIBUTIONS = True       # Enables 3D distribution visualization
```

**Running the Experiment**:
```bash
python experiments/kr_map_dataset_experiment.py
```

**Output**:
- 3D distribution plots with posterior, prior, and KR map results
- Saves full run iterates for video/animation generation
- Results include distribution shift media (if enabled)
- Organised in `results/full_experiment_benchmarks/distribution_plots/`

**Extending to Other Datasets**:
Replace the CSV paths:
```python
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "your_dataset_folder")
PRIOR_CSV_PATH = os.path.join(DATASET_DIR, "prior.csv")
POSTERIOR_CSV_PATH = os.path.join(DATASET_DIR, "posterior.csv")
```

---

### 3. Solver Runtime Scaling Experiment (`experiments/solver_runtime_scaling_experiment.py`)

**Purpose**: Benchmark solver runtime across multiple problem dimensions and seeds, comparing Dykstra variants with QP backends.

**Sweep Parameters**:
```python
DIMENSIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50]
SEEDS = [111, 222, 333]         # Three random seeds for stability
NUM_PARTICLES = 500             # Fixed across all runs
```

**Solvers Compared**:
- `fast_dykstra`: Fast-forward Dykstra (default/primary)
- `vanilla_dykstra`: Standard Dykstra algorithm
- QP backends (CVXOPT, Gurobi, scipy.optimize, quadprog) - some suppressed via `SUPPRESSED_SOLVER_LABELS`

**Output**: Comprehensive benchmark tables and plots
- **Raw component CSVs**: Per-solver, per-component runtimes
- **Aggregated CSVs**: Mean/std/min/max across seeds
- **Plots**: Runtime vs dimension for both full-map and per-component metrics
- **Summary JSON**: Metadata and artifact locations

**Running the Experiment**:
```bash
python experiments/solver_runtime_scaling_experiment.py
```

**Output Location**:
```
results/data_generation/latex_visual/
├── solver_runtime_scaling_component_raw_TS=*.csv
├── solver_runtime_scaling_component_agg_TS=*.csv
├── solver_runtime_scaling_fullmap_raw_TS=*.csv
├── solver_runtime_scaling_fullmap_agg_TS=*.csv
├── solver_runtime_vs_dimension_component_TS=*.png
├── solver_runtime_vs_dimension_fullmap_TS=*.png
└── solver_runtime_scaling_summary_TS=*.json
```

**Customisation**:
- Adjust dimension sweep range: modify `DIMENSIONS` list
- Change number of seeds: modify `SEEDS` list
- Suppress specific solvers: add to `SUPPRESSED_SOLVER_LABELS`
- Modify data generator complexity: change `LayeredBoomerangShearFunction` parameters

---

## Common Configuration Options

All experiments use a shared configuration system. Key parameters:

### Optimisation Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `MAX_OUTER_ITER` | Outer gradient descent iterations | 17500 |
| `LEARNING_RATE` | Initial SGD step size | 0.07-0.1 |
| `LR_DECAY` | Learning rate decay (0 = no decay) | 0.01 |
| `L1_REG` | L1 regularisation on weights | 0.0 |
| `GRADIENT_CLIP_VALUE` | Gradient clipping threshold | 10.0 |

### Dykstra Projection Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `BASE_INNER_ITER` | Min inner iterations per outer iteration | 1 |
| `MAX_INNER_ITERS` | Max inner iterations | 5-10 |
| `INEXACT_POWER` | Exponent for inexact projection scaling | computed |

### SGD Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `BATCH_SIZE` | Batch size (None = full-batch) | 700 |
| `RNG_SEED` | Random seed for batching | SEED + 1 |

### Iterative Hard Thresholding (IHT)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `PRUNE_THRESHOLD` | Weight pruning threshold | 0.01 |
| `PRUNE_INTERVAL` | Iterations between pruning | 100 |

### Plot Parameters
| Parameter | Description |
|-----------|-------------|
| `PLOT_DISTRIBUTIONS` | Enable distribution visualisation |
| `PLOT_DYKSTRA_ITERATES` | Show Dykstra projection convergence |
| `X_LIM`, `Y_LIM` | Plot axis limits |

### Run Modes
| Mode | Purpose |
|------|---------|
| `"fast"` | Run only fast-forward Dykstra |
| `"vanilla"` | Run only vanilla Dykstra |
| `"both"` | Run both Dykstra variants for comparison |
| `"benchmark"` | Run all registered solvers (QP + Dykstra) |

---

## Project Structure

```
.
├── experiments/                  # Main experiment entry points
│   ├── kr_map_experiment.py       # Synthetic data experiments
│   ├── kr_map_dataset_experiment.py # Real dataset experiments
│   └── solver_runtime_scaling_experiment.py # Performance benchmarking
├── core/                         # Core framework
│   ├── config.py                 # Configuration dataclasses
│   ├── data.py                   # Data sources and loaders
│   ├── runner.py                 # Experiment runner and orchestration
│   └── io.py                     # File I/O utilities
├── utils/                        # Utilities and algorithms
│   ├── optimal_transport.py      # KR map and basis functions
│   ├── data_generator.py         # Synthetic data generators
│   ├── pgd_solver.py             # Projected gradient descent solver
│   ├── projection_solver.py      # Dykstra projection implementations
│   ├── hermite.py                # Hermite polynomial computations
│   ├── plotter.py                # Visualization utilities
│   └── projection_result.py      # Result data structures
├── data/                         # Dataset storage
│   └── Lorenz 1963 and Feedback Particle Filter/
│       └── prediction_flow_data/
│           ├── prior.csv
│           └── posterior.csv
├── tests/                        # Unit and integration tests
├── results/                      # Experiment outputs (auto-generated)
└── README.md                     # This file
```

---

## Quick Start

### 1. Run a Simple 2D Synthetic Experiment
```bash
python experiments/kr_map_experiment.py
```
This runs the default BoomerangShearFunction with 2500 particles and generates distribution plots.

### 2. Run on Real Lorenz Data
```bash
python experiments/kr_map_dataset_experiment.py
```
Uses the Lorenz 1963 system data with whitening preprocessing.

### 3. Benchmark Solvers Across Dimensions
```bash
python experiments/solver_runtime_scaling_experiment.py
```
Generates a comprehensive runtime comparison table and plots.

---

## Advanced Usage

### Modifying Data Generators
Edit the `DATA_GENERATOR` assignment in the experiment file:

```python
# Example: RoughLine with custom noise
DATA_GENERATOR = DataGenerator(
    shear_function=RoughLineShearFunction(sigma=0.2),  # Adjust noise
)

# Example: Cubic shear
DATA_GENERATOR = DataGenerator(
    shear_function=CubicShearFunction(strength=0.5),
)
```

### Tuning Optimisation Hyperparameters
Modify module-level constants before running:

```python
MAX_OUTER_ITER = 10000          # Fewer iterations for faster testing
LEARNING_RATE = 0.05            # Lower learning rate
L1_REG = 0.001                  # Add sparsity regularisation
BATCH_SIZE = 200                # Smaller batches
```

### Comparing Solvers
Set `RUN_SOLVER_MODE`:
```python
RUN_SOLVER_MODE = "both"        # Compare vanilla vs fast-forward
RUN_SOLVER_MODE = "benchmark"   # Compare all registered solvers
```

### Customising Visualisations
```python
PLOT_DISTRIBUTIONS = True
PLOT_SIZE = 10.0                # Larger plot range
PLOT_DYKSTRA_ITERATES = True   # Show projection convergence
PLOT_DYKSTRA_OUTER_ITERATIONS = [0, 5000, -1]  # Specific iterations
```

---

## Dependencies

Core dependencies:
- `numpy`: Array computations
- `scipy`: Optimisation and linear algebra
- `matplotlib`: Plotting and visualisation
- Optional QP solvers: `cvxopt`, `gurobi`, `quadprog`

Install via:
```bash
pip install numpy scipy matplotlib
```

---

## Output Interpretation

### Distribution Plots
- **Posterior Distribution**: Target reference distribution
- **Prior Distribution**: Source distribution to be mapped
- **Mapped Prior (Vanilla Dykstra)**: Result using standard Dykstra
- **Mapped Prior (Fast-Forward Dykstra)**: Result using fast-forward variant

### Benchmark Tables
- **Raw CSV**: Individual component solver runtimes
- **Aggregated CSV**: Mean/std/median statistics per dimension-solver pair
- **Plots**: Runtime scaling trends with error bars

### Convergence Metrics
- **Objective Value**: Loss function tracking over iterations
- **L2 Norm of Weights**: Parameter magnitude evolution
- **Solver Runtime**: Total elapsed time for projection steps

---

## Troubleshooting

**Issue**: Experiment runs slowly
- Reduce `NUM_PARTICLES` for testing
- Reduce `MAX_OUTER_ITER` to 5000 for quick runs
- Set `BATCH_SIZE` to enable stochastic updates

**Issue**: Memory usage is high
- Disable `SAVE_FULL_RUN_ITERATES` if not needed
- Disable `SAVE_DISTRIBUTION_SHIFT_MEDIA`
- Reduce particle count

**Issue**: Visualization not appearing
- Set `PLOT_DISTRIBUTIONS = True` and appropriate axis limits
- Check output directory exists: `results/full_experiment_benchmarks/`

---

## Citation

If you use this code in research, please cite the underlying methods and datasets used.

---

## Notes

- Experiments save results with timestamps to avoid overwriting
- Random seeds ensure reproducibility
- Dataset experiments require the Lorenz data CSV files in `data/` folder
- QP solver availability depends on installed packages (optional)
