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

*   **`core/`**: Essential logic including `detectors` (boundary monitoring), `engines` (Qiskit integration), `models` (Dirac circuits), and `operators`.
*   **`utils/io/`**: Data management, supporting `.h5` format for high-fidelity simulation results.
*   **`utils/runtime/`**: Execution scripts for running simulations and generating visualizations.
*   **`configs/`**: YAML-based configuration management for reproducible experiments.
*   **`tests/`**: Comprehensive suite of unit tests for framework validation.

## Installation

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
    *   `true`: Enables **warmup mode**, setting the threshold based on `p_min`.
    *   `false`: Uses the static value defined in `lattice.threshold`.
*   **`lattice.threshold`**: Sensitivity value for boundary detection.

#### Validity Control
*   **`validity.stricted`**: Defines the simulation's rigor regarding safety limits.
    *   `true`: Immediately **interrupts** the current simulation if `n_safe < p_min` and proceeds to the next configuration.
    *   `false`: Logs a violation warning but allows the execution to continue until completion.

### Advanced Usage Examples

To run simulations with different validation behaviors, point to your specific configuration files:

```bash
# Running with strict validation (halts on instability)
python -m bav_dqs.utils.runtime.run_dirac_simulation --config src/configs/strict_validation.yaml

# Running with automatic threshold (warmup mode enabled)
python -m bav_dqs.utils.runtime.run_dirac_simulation --config src/configs/auto_threshold_setup.yaml
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

<a href="https://embits/digital" target="_blank"><strong>EMBITS.DIGITAL</strong></a>, Brazil

Email: elionai@embits.digital