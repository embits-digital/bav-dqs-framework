"""Numerical check of the Trotter order of the BAV-DQS step circuit and of Richardson pairing.

Uses the framework-free reference model in ``bav_dqs.core.models.dirac_numpy`` (a NumPy replica of
``dirac_circuits.build_step_circuit`` and the exact propagator of

    H = m * sum_q (-1)^q Z_q  +  w * sum_q (X_q X_{q+1} + Y_q Y_{q+1}) ).

Because H conserves the number of excitations and the initial state has a single excitation, the dynamics
lives in an N-dimensional subspace, so the exact propagator is a small ``eigh`` and lattices up to N=20 are
trivial. ``--check-full-space`` validates this reduction against a brute-force 2^N simulation of the gate
sequence (N <= 8). The replica is also compared with the real Qiskit circuit in ``tests/test_trotter_order.py``.

For each time step dt the script reports, at fixed physical times, the error of the occupations n_i(t)
against the exact propagator, the observed convergence order, and the error of the Richardson combination
(2^p n_half - n_full) / (2^p - 1) for p = 1 and p = 2.

Usage:
    python -m bav_dqs.utils.runtime.verify_trotter_order --n 6 8 12 --scheme lie strang
"""
from __future__ import annotations
import argparse
import json
from typing import Dict, List

import numpy as np

from bav_dqs.core.models.dirac_geometry import source_index
from bav_dqs.core.models.dirac_numpy import (  # noqa: F401  (re-exported for tests/tools)
    SCHEMES, evolve, exact_propagator, hamiltonian_1exc, step_lie, step_strang,
)


# ------------------------------------------------------------------ full-space cross-check
def check_full_space(n: int, m: float, w: float, dt: float, steps: int = 7) -> float:
    """Max |P_full - P_1exc| after `steps` steps of the real gate sequence in the 2^N space."""
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1.0, -1.0]).astype(complex)
    I2 = np.eye(2, dtype=complex)

    def op(single: Dict[int, np.ndarray]) -> np.ndarray:
        out = np.array([[1.0 + 0j]])
        for q in range(n):
            out = np.kron(out, single.get(q, I2))
        return out

    def rot(theta: float, P: np.ndarray) -> np.ndarray:  # exp(-i theta/2 P)
        return np.cos(theta / 2) * np.eye(P.shape[0]) - 1j * np.sin(theta / 2) * P

    U = np.eye(2 ** n, dtype=complex)
    for q in range(n):
        sign = 1.0 if q % 2 == 0 else -1.0
        U = rot(sign * 2 * m * dt, op({q: Z})) @ U
    for q in range(n - 1):
        U = rot(2 * w * dt, op({q: X, q + 1: X})) @ U
        U = rot(2 * w * dt, op({q: Y, q + 1: Y})) @ U

    c = source_index(n)
    psi = np.zeros(2 ** n, complex)
    psi[1 << (n - 1 - c)] = 1.0  # qubit c (tensor position c) in |1>
    for _ in range(steps):
        psi = U @ psi
    probs = np.abs(psi) ** 2
    p_full = np.array([sum(probs[k] for k in range(2 ** n) if (k >> (n - 1 - i)) & 1) for i in range(n)])

    psi1 = np.zeros(n, complex)
    psi1[c] = 1.0
    p_red = np.abs(evolve(step_lie(n, m, w, dt), psi1, steps)) ** 2
    return float(np.max(np.abs(p_full - p_red)))


# ----------------------------------------------------------------------------- analysis
def occupations(scheme: str, n: int, m: float, w: float, dt: float, t: float) -> np.ndarray:
    steps = int(round(t / dt))
    assert abs(steps * dt - t) < 1e-9, f"t={t} is not a multiple of dt={dt}"
    psi0 = np.zeros(n, complex)
    psi0[source_index(n)] = 1.0
    return np.abs(evolve(SCHEMES[scheme](n, m, w, dt), psi0, steps)) ** 2


def exact_occupations(n: int, m: float, w: float, t: float) -> np.ndarray:
    psi0 = np.zeros(n, complex)
    psi0[source_index(n)] = 1.0
    return np.abs(exact_propagator(hamiltonian_1exc(n, m, w), t) @ psi0) ** 2


def richardson(full: np.ndarray, half: np.ndarray, p: int) -> np.ndarray:
    f = 2.0 ** p
    return (f * half - full) / (f - 1.0)


def run(n: int, scheme: str, m: float, w: float, dts: List[float], times: List[float]) -> Dict:
    out = {"n": n, "scheme": scheme, "rows": []}
    for t in times:
        ex = exact_occupations(n, m, w, t)
        errs = {}
        for dt in dts:
            occ = occupations(scheme, n, m, w, dt, t)
            errs[dt] = float(np.max(np.abs(occ - ex)))
        orders = {dts[i]: float(np.log2(errs[dts[i]] / errs[dts[i + 1]])) for i in range(len(dts) - 1)}
        rich = {}
        for dt in dts[:-1]:  # full = dt, half = dt/2 (next entry)
            full = occupations(scheme, n, m, w, dt, t)
            half = occupations(scheme, n, m, w, dt / 2, t)
            rich[dt] = {
                "full": float(np.max(np.abs(full - ex))),
                "half": float(np.max(np.abs(half - ex))),
                "rich_p1": float(np.max(np.abs(richardson(full, half, 1) - ex))),
                "rich_p2": float(np.max(np.abs(richardson(full, half, 2) - ex))),
            }
        out["rows"].append({"t": t, "occ_err": {str(k): v for k, v in errs.items()},
                            "observed_order": {str(k): v for k, v in orders.items()},
                            "richardson": {str(k): v for k, v in rich.items()}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[6, 8, 12])
    ap.add_argument("--scheme", nargs="+", choices=list(SCHEMES), default=["lie", "strang"])
    ap.add_argument("--m", type=float, default=0.5)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--dts", type=float, nargs="+", default=[0.04, 0.02, 0.01, 0.005, 0.0025])
    ap.add_argument("--times", type=float, nargs="+", default=[0.32, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--check-full-space", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    if a.check_full_space:
        for n in [x for x in a.n if x <= 8]:
            print(f"[full-space check] N={n}: max |P_2^N - P_1exc| = {check_full_space(n, a.m, a.w, 0.02):.3e}")

    results = []
    for n in a.n:
        for scheme in a.scheme:
            r = run(n, scheme, a.m, a.w, a.dts, a.times)
            results.append(r)
            print(f"\n=== N={n}  scheme={scheme}  (m={a.m}, w={a.w}) ===")
            print("max_i |n_i - n_i^exact|  at fixed physical time; observed order = log2(err(dt)/err(dt/2))")
            for row in r["rows"]:
                errs = "  ".join(f"{float(k):<7g}:{v:9.2e}" for k, v in row["occ_err"].items())
                ords = " ".join(f"{v:5.2f}" for v in row["observed_order"].values())
                print(f"t={row['t']:<5g} {errs}\n        order: {ords}")
            print("Richardson at dt_full=0.02 / dt_half=0.01 (error vs exact):")
            for row in r["rows"]:
                x = row["richardson"].get("0.02")
                if x:
                    print(f"  t={row['t']:<5g} full={x['full']:.2e} half={x['half']:.2e} "
                          f"rich(p=1)={x['rich_p1']:.2e} rich(p=2)={x['rich_p2']:.2e}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
