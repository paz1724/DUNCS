# DUNCS — Project Memory for Claude

## Project Overview

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


**DUNCS** (Deep Unfolded Sparse Covariance ADMM for DoA Recovery) is a Python research framework for **Direction of Arrival (DoA) estimation** using deep-unfolding techniques applied to sparse arrays. It implements and compares multiple deep learning models against classical subspace-based methods.

The two primary models are:
- **DUNCS** — Deep-unfolded ADMM network for sparse covariance recovery, followed by a subspace method (e.g., ESPRIT).
- **SparseNet** — A sparse-array-aware neural network using differentiable subspace processing.

---

## Repository Structure

```
DUNCS/
├── main.py                  # Entry point — run a single model via config
├── run_simulation.py        # SimulationRunner class orchestrating train/eval
├── compare_models.py        # Trains DUNCS & SparseNet side-by-side and compares
├── train_single_model.py    # Standalone single-model training script
├── models_config.json       # JSON model config (used by train_single_model.py)
├── requirements.txt         # Python dependencies
├── run_comparison.m         # MATLAB script for running comparison plots
├── plot_comparison.m        # MATLAB script for generating plots
├── data/                    # Auto-generated: datasets, weights, simulation results
│   ├── datasets/
│   ├── weights/
│   └── simulations/
└── src/
    ├── config/
    │   ├── DUNCS.yaml           # Config for DUNCS model
    │   ├── sparseNet.yaml       # Config for SparseNet model
    │   └── simulation_config.py # Dataclasses for config parsing (OmegaConf)
    ├── models.py                # ModelGenerator factory class
    ├── system_model.py          # SystemModel (array geometry, steering vectors)
    ├── signal_creation.py       # Samples class for dataset generation
    ├── data_handler.py          # Dataset creation/loading, DataLoader utilities
    ├── training.py              # Training loop, TrainingParams, schedulers
    ├── evaluation.py            # Evaluation loop for DNN and model-based methods
    ├── methods.py               # High-level method dispatch
    ├── plotting.py              # Plot utilities
    ├── utils.py                 # Device detection, misc helpers
    ├── sparse_array.py          # Sparse array geometry utilities (MRA, co-array)
    ├── steering_vector_generator.py  # Singleton steering vector generator
    ├── models_pack/             # Individual model implementations
    │   ├── sparse_cov_admm_unfold.py  # DUNCS model
    │   ├── sparse_net.py              # SparseNet model
    │   ├── subspacenet.py             # SubspaceNet model
    │   ├── dcd_music.py               # DCD-MUSIC model
    │   ├── deep_augmented_music.py    # DA-MUSIC model
    │   ├── deep_cnn.py                # DeepCNN model
    │   ├── deep_root_music.py         # Deep Root-MUSIC model
    │   ├── trans_music.py             # TransMUSIC model
    │   └── parent_model.py            # Base model class
    ├── methods_pack/            # Classical subspace method implementations
    │   ├── music.py                   # MUSIC algorithm
    │   ├── esprit.py                  # ESPRIT algorithm
    │   ├── root_music.py              # Root-MUSIC algorithm
    │   ├── cov_reconstruct.py         # Covariance reconstruction (ADMM, averaging, etc.)
    │   └── subspace_method.py         # Base subspace method
    └── metrics/
        ├── criterions.py              # Loss functions (RMSPE, ADMM objective, etc.)
        └── crb.py                     # Cramér–Rao Bound calculations
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+ (uses `match`/`case`). PyTorch 2.6 with CUDA recommended.

### 2. Run a Single Simulation

```bash
# Run DUNCS model
python main.py duncs

# Run SparseNet model
python main.py sparsenet
```

The entry point reads the corresponding YAML config and launches `SimulationRunner.run()`.

### 3. Compare DUNCS vs SparseNet

```bash
python compare_models.py
```

Trains both models on the same shared dataset, produces comparison training curves, and a final bar chart of test RMSPE. Outputs to `data/simulations/results/plots/`.

### 4. Train a Single Model (standalone)

```bash
python train_single_model.py
```

Uses `models_config.json` for configuration.

---

## Configuration (YAML)

Configs live in `src/config/DUNCS.yaml` and `src/config/sparseNet.yaml`. Key sections:

| Section | Key Parameters |
|---|---|
| `system_model` | `N` (sensors), `M` (sources), `T` (snapshots), `snr`, `array_form` (`ula` / `mra-N`), `field_type` (`Far`/`Near`), `signal_nature` |
| `admm_params` | `mu`, `rho` — ADMM penalty and step-size parameters |
| `model` | `model_type` (e.g. `DUNCS`, `SparseNet`), `model_params` (model-specific) |
| `training` | `samples_size`, `epochs`, `batch_size`, `optimizer`, `learning_rate`, `scheduler` |
| `evaluation` | `criterion` (`rmspe`), `covariance_reconstruction`, `subspace_methods` |
| `commands` | `create_data`, `train_model`, `evaluate_mode`, `save_model`, `save_plots` |
| `scenario` | Optional sweep over parameters (e.g. `snr`, `T`) for batch simulations |

### DUNCS-Specific Model Params
```yaml
model_params:
  num_iterations: 20       # Number of unrolled ADMM iterations
  subspace_method: "esprit"  # esprit | music | root_music
  criterion: "rmspe"
  mu: 2.5e-3
  rho: 2
```

### SparseNet-Specific Model Params
```yaml
model_params:
  tau: 8                   # Max autocorrelation lag
  diff_method: "esprit"    # Differentiable subspace method
```

---

## Available Models (`model_type` string keys)

| Key | Class | File |
|---|---|---|
| `DUNCS` | `DUNCS` | `models_pack/sparse_cov_admm_unfold.py` |
| `SparseNet` | `SparseNet` | `models_pack/sparse_net.py` |
| `SubspaceNet` | `SubspaceNet` | `models_pack/subspacenet.py` |
| `DCDMUSIC` | `DCDMUSIC` | `models_pack/dcd_music.py` |
| `DA-MUSIC` | `DeepAugmentedMUSIC` | `models_pack/deep_augmented_music.py` |
| `DeepCNN` | `DeepCNN` | `models_pack/deep_cnn.py` |
| `DR_MUSIC` | `DeepRootMUSIC` | `models_pack/deep_root_music.py` |
| `TransMUSIC` | `TransMUSIC` | `models_pack/trans_music.py` |

Use `ModelGenerator` (in `src/models.py`) to instantiate any model programmatically:
```python
model_gen = (
    ModelGenerator()
    .set_model_type("DUNCS")
    .set_system_model(system_model)
    .set_model_params(config.model.model_params)
    .set_model()
)
model = model_gen.model
```

---

## Key Classes & Entry Points

| Class / Function | Location | Purpose |
|---|---|---|
| `SimulationRunner` | `run_simulation.py` | Top-level orchestrator (create data → train → evaluate) |
| `SystemModel` | `src/system_model.py` | Array geometry, steering vectors, Fresnel/Fraunhofer distances |
| `Samples` | `src/signal_creation.py` | Generates training/test signal samples |
| `ModelGenerator` | `src/models.py` | Factory for instantiating any model |
| `TrainingParams` | `src/training.py` | Builder for training configuration |
| `train()` | `src/training.py` | Training loop |
| `evaluate()` | `src/evaluation.py` | Full evaluation (DNN + model-based + classical baselines) |
| `create_dataset()` | `src/data_handler.py` | Dataset generation |
| `load_simulation_config()` | `src/config/simulation_config.py` | Parses YAML config into dataclass |

---

## Array Geometry

- **ULA** (Uniform Linear Array): Standard, N sensors evenly spaced at λ/2.
- **MRA-N** (Minimum Redundancy Array): Sparse array with N physical elements; virtual ULA is computed via the difference co-array using `sparse_array.py`.

---

## Evaluation Metrics

- **RMSPE** — Root Mean Squared Phase Error (primary metric).
- **ADMM Objective** — Alternative criterion based on the ADMM optimization objective.
- **Cramér–Rao Bound (CRB)** — Lower bound on estimation variance, computed in `src/metrics/crb.py`.

Classical baselines evaluated include: **ESPRIT**, **MUSIC**, **Root-MUSIC**.

---

## Data & Artifacts

Generated automatically at runtime under `data/`:
- `data/datasets/` — Saved training/test datasets (if `save_dataset: true`).
- `data/weights/` — Intermediate and final model weights (if `save_model: true`).
- `data/simulations/results/` — Score logs and comparison plots.

---

## Notes

- Uses **OmegaConf** for YAML config parsing with variable interpolation (e.g., `${admm_params.mu}`).
- `SteeringVectorGenerator` is a **singleton** — call `SteeringVectorGenerator.reset_instance()` between simulations with different system model configurations.
- Antenna pattern support: if `antenna_pattern: true` in config, loads a `.mat` file containing a complex phasor matrix `A [Nelements, Nazimuth, Nfreqs]` from an `sSteering` struct.
- Device auto-detected via `src/utils.py` (`device`).
