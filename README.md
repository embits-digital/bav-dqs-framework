# Boundary-Aware Validity Framework for Digital Quantum Simulation



<a href="https://orcid.org/0000-0003-4605-1949" target="_blank"><img src="https://img.shields.io/badge/orcid-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" /></a> &nbsp; &nbsp; 
<a href="https://opensource.org/license/mit" target="_blank"><img src="https://img.shields.io/badge/MIT-red?style=for-the-badge" /></a> &nbsp; &nbsp; 
<a href="https://doi.org/10.5281/zenodo.18918489" target="_blank"><img src="https://img.shields.io/badge/DOI-FAB70C?style=for-the-badge" /></a> &nbsp; &nbsp; 
<a href="https://zenodo.org/records/18918490" target="_blank"><img src="https://img.shields.io/badge/ZENODO-blue?style=for-the-badge" /></a>

This repository contains the official implementation of the **Boundary-Aware Validity (BAV)** framework. It provides a methodological layer to ensure inferential admissibility in finite-domain digital quantum simulations by identifying causality-safe temporal windows.

## Abstract

Digital quantum simulations are necessarily implemented in finite spatial domains and evaluated over finite temporal windows. While substantial effort has been devoted to controlling algorithmic error and hardware noise, the prior question of inferential admissibility remains under-formalized: when do time-resolved observables remain causally isolated from boundary effects such that spectral conclusions are physically meaningful? 

We introduce a boundary-aware validity framework that establishes an operational criterion for **inferential gating**. The framework identifies measurable boundary markers, determines a causality-safe temporal window, and restricts spectral analysis to data supported by verified causal isolation. We demonstrate the framework in a lattice simulation of relativistic wavepacket dynamics.

## Keywords

*   **Primary:** Digital Quantum Simulation • Boundary-Aware Validity • Causal Isolation

*   **Technical:** Lattice Field Theory • Spectral Inference • Finite-Domain Dynamics • Inferential Admissibility • Zitterbewegung


## Repository Structure

The project is structured as a modular Python package located in `src/bav_dqs`:

*   **`core/`**: Essential logic including `detectors` (boundary monitoring), `engines` (Qiskit integration), `models` (Dirac circuits), and `operators`.
*   **`io/`**: Data management, supporting `.h5` format for high-fidelity simulation results.
*   **`runtime/`**: Execution scripts for running simulations and generating visualizations.
*   **`configs/`**: YAML-based configuration management for reproducible experiments.
*   **`tests/`**: Comprehensive suite of unit tests for framework validation.

## Installation

The project uses `pyproject.toml` for dependency management. To install in editable mode:

```bash
git clone https://github.com
cd bav-dqs-framework
pip install -e .
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

### 1. Running a Simulation
To execute a Dirac wavepacket simulation using a configuration file:

```bash
python -m bav_dqs.runtime.run_dirac_simulation --config src/configs/dirac_simulation.yaml --results-dir ./results
```

### 2. Generating Plot and Analysis
To execute a Dirac wavepacket simulation using a configuration file:

```bash
python -m bav_dqs.runtime.generate_plots --data-file=results/dirac_simulation_20260309_020940.h5
```


## Citation
If you use this framework or the associated data in your research, please cite:

### Paper:
Cordeiro, E. M. (2026). Boundary-Aware Validity Framework for Digital Quantum Simulation.

### Data/Software:
E. Moura Cordeiro, “[Boundary-Aware Validity Framework for Digital Quantum Simulation](https://doi.org/10.5281/zenodo.18918489)”. Zenodo, Mar. 09, 2026. doi: 10.5281/zenodo.18918490.


## Contact
Elionai Moura Cordeiro

<a href="https://embits/digital" target="_blank"><strong>EMBITS.DIGITAL</strong></a>, Brazil

Email: elionai@embits.digital