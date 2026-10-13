# Boundary-Aware Validity Framework for Digital Quantum Simulation



<a href="https://orcid.org/0000-0003-4605-1949" target="_blank"><img src="https://img.shields.io/badge/orcid-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" /></a> &nbsp; &nbsp; 
<a href="https://opensource.org/license/mit" target="_blank"><img src="https://img.shields.io/badge/MIT-red?style=for-the-badge" /></a> &nbsp; &nbsp; 
<a href="https://doi.org/10.5281/zenodo.18974237" target="_blank"><img src="https://img.shields.io/badge/DOI-FAB70C?style=for-the-badge" /></a> &nbsp; &nbsp; 
<a href="https://doi.org/10.5281/zenodo.18974237" target="_blank"><img src="https://img.shields.io/badge/ZENODO-blue?style=for-the-badge" /></a>

This repository contains the official implementation of the **Boundary-Aware Validity (BAV)** framework. It provides a methodological layer to ensure inferential admissibility in finite-domain digital quantum simulations by identifying causality-safe temporal windows.

## Abstract

Digital quantum simulations are necessarily implemented in finite spatial domains and evaluated over finite temporal windows. While substantial effort has been devoted to controlling algorithmic error and hardware noise, the prior question of inferential admissibility remains under-formalized: when do time-resolved observables remain causally isolated from boundary effects such that spectral conclusions are physically meaningful? 

We introduce a boundary-aware validity framework that establishes an operational criterion for **inferential gating**. The framework identifies measurable boundary markers, determines a causality-safe temporal window, and restricts spectral analysis to data supported by verified causal isolation. We demonstrate the framework in a lattice simulation of relativistic wavepacket dynamics.

## Keywords

*   **Primary:** Digital Quantum Simulation • Boundary-Aware Validity • Causal Isolation

*   **Technical:** Lattice Field Theory • Spectral Inference • Finite-Domain Dynamics • Inferential Admissibility • Zitterbewegung


## Repository Structure

The source code is available under MIT license at [https://github.com/embits-digital/bav-dqs-framework](https://github.com/embits-digital/bav-dqs-framework). The project is structured as a modular Python package located in `src/bav_dqs`:

*   **`core/`**: Essential logic including `detectors` (boundary monitoring, use-case agnostic), `engines` (the `BaseEngine` contract plus the Qiskit and NumPy connectors), `models` (use-case dynamics, e.g. Dirac), `operators`, and `connectors.py` (lazy registry that maps `backend.framework` to a connector and a model builder, so no quantum framework is imported unless selected).
*   **`utils/io/`**: Data management, supporting `.h5` format for high-fidelity simulation results.
*   **`utils/runtime/`**: Execution scripts for running simulations and generating visualizations.
*   **`configs/`**: YAML-based configuration management for reproducible experiments.
*   **`tests/`**: Comprehensive suite of unit tests for framework validation.

## Installation

**Requirements:** Python **>= 3.11** (minimum imposed by `pandas>=3.0`). The test suite is verified on Python 3.11 and 3.14 with `qiskit==2.3.0` and `qiskit-aer==0.17.2`. The preprint v1 results were produced with Python 3.11.

The project uses `pyproject.toml` for dependency management. To install in editable mode:

```bash
git clone https://github.com/embits-digital/bav-dqs-framework
cd bav-dqs-framework
pip install -e .
```

### TL;DR
To generate report from sample data (similar to dirac simulations use case), just type:
```bash
python -m bav_dqs.utils.runtime.generate_plots --data_file=sample\dirac_simulation\dirac_simulation_20260312_175120.h5 --config=sample\dirac_simulation\dirac_simulation.yaml
```

### Development Setup

To contribute to the project or run the tests, install the development dependencies using the [dev] selector:

```
pip install -e ".[dev]"
```

To run the default test suit:

```
python -m pytest
```

## Usage

### 1. Running a Simulation: Dirac Simulation Use-Case
To execute a Dirac wavepacket simulation using a configuration file:

```bash
python -m bav_dqs.utils.runtime.run_dirac_simulation --config src/configs/dirac_simulation.yaml --results-dir ./results
```

### 2. Generating Plot and Analysis
Use the generate_plots module to analyze existing results. Replace the --data_file path with your generated .h5 file

```bash
python -m bav_dqs.utils.runtime.generate_plots --data_file=results/dirac_simulation_20260309_020940.h5
```

If you used a custom configuration file, you can specify the same settings to generate plots and tables based on the YAML file.

```bash
python -m bav_dqs.utils.runtime.generate_plots --data_file=test_results\dirac_simulation_20260310_201229.h5 --config src/configs/dirac_simulation.yaml
```

The simulation behavior is controlled via the `src/configs/dirac_simulation.yaml` file. Below are the key parameters for precision adjustment and validity control:

#### Lattice and Threshold Parameters
*   **`lattice.auto_threshold`**: Determines if the system calculates the threshold automatically.
    *   `true`: Enables **warmup mode**: the threshold is calibrated from the first `p_min` steps (mean + 5 sigma, never below `lattice.threshold`) and then applied to the detector.
    *   `false`: Uses the static value defined in `lattice.threshold`.
*   **`lattice.threshold`**: Sensitivity value for boundary detection.
*   **`lattice.edge_window`** (W): Number of outermost sites monitored on each edge. The boundary marker is the **maximum per-site deviation from the t=0 profile** inside the window, `max_i |n_i(t) - n_i(0)|` (peak criterion, deliberately more conservative than a window average: `mean <= max <= W * mean`). The windows must not contain sites that are non-zero at t=0 by construction (the use case declares them; for the Dirac lattice that is the source site `n // 2`, so the maximum valid W is `min(n//2, n-1-n//2)`, e.g. W=3 requires N >= 8). The detector itself is use-case agnostic: it only receives the declared sites through `BoundaryDetector.validate_edge_windows`. Widths that violate this are skipped with a warning by the runner and rejected with a `ValueError` by `run_boundary_detection`.
*   **`backend.framework`** (default `qiskit`): quantum framework behind the engine, resolved through the connector registry (`bav_dqs/core/connectors.py`) and imported lazily. Available: `qiskit`, and `numpy` (exact, noiseless reference engine in the single-excitation subspace; needs no quantum framework). The detector, the pipeline and the reports are identical for every framework; supporting a new one means registering a connector and a model builder.
*   **`lattice.edge_persistence`**: Consecutive steps with deviation `>= threshold` needed to confirm a hit. The reported hit index is `step - edge_persistence`.

Every run stores `framework`, `edge_window`, `edge_persistence`, `richardson_order_p`, `integrator_order`, `richardson_order_consistent`, `auto_threshold`, `threshold_configured`, `p_min`, `d_hit_eff` and `edge_marker` as HDF5 attributes, and the file root stores the full resolved config, its SHA-256 and the Git commit (`git_commit`, `git_dirty`). A warning is emitted when `richardson.order_p` differs from the integrator's global error order. Use `python -m bav_dqs.utils.runtime.make_manifest --data_file <file.h5> --paper-ref "Table I" --category official` to produce a `provenance.json` for a Zenodo deposit (see `docs/ZENODO_DEPOSITS.md`).

**Time grids and Richardson (notation).** Every width is simulated on two grids: the base grid with step Δt, and the refined grid with step Δt/2 (twice as many steps, same physical horizon). Hits from both grids are compared in physical time; a refined-grid hit is mapped to the base grid with the floor (never discarded). Richardson combines the two grids as `(2^p · f(Δt/2) − f(Δt)) / (2^p − 1)`, where `richardson.order_p` = p is the **global error order of the integrator** (not a number of resolutions): p = 1 for the first-order Lie-Trotter circuit. Richardson is an optional complement; detection and the gate never depend on it.

**`richardson.order_p` rules.** When `richardson.enabled: true`, `order_p` is **mandatory** and must be an integer >= 1 (no default is ever assumed; `0`, `1.5` or a missing key abort the run). It is read once (`parse_richardson_cfg`) and that same value is used for the extrapolation and written to the HDF5 attribute `richardson_order_p`. A value different from the integrator order (`integrator_order`, currently 1) is allowed, e.g. to reproduce the v1 runs with p = 2, but emits a warning and is recorded as `richardson_order_consistent = false`. With Richardson disabled, `richardson_order_p = -1` (no extrapolation computed) and `richardson_order_consistent = true` (no inconsistent extrapolation applied).

**Apparent boundary-arrival velocity.** The report table `causal_velocity.tex` lists `v_hit = d_eff / t_hit`, where `d_eff` is the saved distance from the source to the window site of the side that was hit (for even N the right edge is one site closer than the left one) and `t_hit = p_hit * dt`. It depends on the threshold and on the persistence rule and is **not** a Lieb-Robinson velocity; legacy files without `d_hit_eff` show `-`.

> The dataset previously named `metric_rich_*` is now `metric_res_gap_*`: it is the resolution gap `|mean_W(full) - mean_W(half)|`, not the boundary marker. Bump `experiment.schema_version` in your YAML when generating new results.

#### Validity Control
*   **`validity.stricted`**: Defines the simulation's rigor regarding safety limits.
    *   `true`: Immediately **interrupts** the current simulation if `n_safe < p_min` and proceeds to the next configuration.
    *   `false`: Logs a violation warning but allows the execution to continue until completion.

### Advanced Usage Examples

To run simulations with different validation behaviors, copy `src/configs/dirac_simulation.yaml` and change the keys described above (the behaviors are controlled only by configuration; no other file is needed):

```bash
# Strict validation (halts a width when n_safe < p_min): set validity.stricted: true in your copy
python -m bav_dqs.utils.runtime.run_dirac_simulation --config path/to/your_strict_config.yaml

# Automatic threshold (warmup mode): set lattice.auto_threshold: true in your copy
python -m bav_dqs.utils.runtime.run_dirac_simulation --config path/to/your_auto_threshold_config.yaml

# Framework-free run (no Qiskit installed): set backend.framework: numpy in your copy
python -m bav_dqs.utils.runtime.run_dirac_simulation --config path/to/your_numpy_config.yaml
```

## Citation

If you use this framework or the associated data in your research, please cite:

### Paper:
Cordeiro, E. M. (2026). Boundary-Aware Validity Framework for Digital Quantum Simulation.

#### How to help:
If you are a registered endorser on ArXiv for nlin.CD (quantum.ph), you can support this submission via the link or code below:

👉 Endorsement Link: https://arxiv.org/auth/endorse?x=49UWOT
👉 Endorsement Code: 49UWOT

If you are not an endorser but know someone in the field of Chaotic Dynamics or Quantum Simulation, I would greatly appreciate a tag or a share.

### Data and Software:
E. Moura Cordeiro, “[Boundary-Aware Validity Framework for Digital Quantum Simulation](https://doi.org/10.5281/zenodo.18974237)”. Zenodo, Mar. 12, 2026. doi: 10.5281/zenodo.18974237.


## Contact
Elionai Moura Cordeiro

<a href="https://embits.digital" target="_blank"><strong>EMBITS.DIGITAL</strong></a>, Brazil

Email: elionai@embits.digital