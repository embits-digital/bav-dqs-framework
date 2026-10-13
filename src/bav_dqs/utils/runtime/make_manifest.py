"""Build a provenance manifest (file -> config -> commit -> paper table/figure) for HDF5 results.

Usage:
    python -m bav_dqs.utils.runtime.make_manifest \
        --data_file results/dirac_simulation_X.h5 --paper-ref "Table I" --category official \
        --out provenance.json

Pass ``--data_file`` multiple times to describe several files in one manifest.
Intended to accompany each Zenodo deposit (official dataset / exploratory material).
"""
from __future__ import annotations
import argparse
import datetime as _date
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bav_dqs.utils.helpers.provenance import sha256_file

CATEGORIES = ("official", "exploratory")

RUN_ATTRS_OF_INTEREST = (
    "n_qubits", "m", "w", "dt_full", "dt_half", "max_steps_full", "threshold",
    "threshold_configured", "edge_marker", "edge_window", "edge_persistence",
    "auto_threshold", "richardson_enabled", "richardson_order_p", "p_min",
    "first_hit_full", "first_hit_half", "first_hit_side", "d_hit_eff",
)


def _to_jsonable(v: Any) -> Any:
    """Converts HDF5/NumPy attribute values into JSON-serializable Python values."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float) and v != v:  # NaN is not valid JSON
        return None
    return v


def describe_file(h5_path: Path, paper_ref: Optional[str], category: Optional[str]) -> Dict[str, Any]:
    import h5py  # imported lazily: only needed when reading real result files

    h5_path = Path(h5_path)
    entry: Dict[str, Any] = {
        "file": h5_path.name,
        "sha256": sha256_file(h5_path),
        "size_bytes": h5_path.stat().st_size,
        "paper_ref": paper_ref,
        "category": category,
    }
    with h5py.File(h5_path, "r") as f:
        entry["global_attrs"] = {k: _to_jsonable(v) for k, v in f.attrs.items()}
        runs: List[Dict[str, Any]] = []
        group = f.get("dirac_simulation/runs")
        if group is not None:
            for run_id in sorted(group.keys()):
                a = group[run_id].attrs
                runs.append({
                    "run_id": run_id,
                    **{k: _to_jsonable(a[k]) for k in RUN_ATTRS_OF_INTEREST if k in a},
                })
        entry["runs"] = runs
    # Files written before provenance capture lack these fields: say so explicitly.
    entry["provenance_complete"] = all(
        k in entry["global_attrs"] for k in ("config_sha256", "git_commit")
    ) and all("edge_window" in r for r in entry["runs"])
    return entry


def build_manifest(files: List[Path], paper_ref: Optional[str], category: Optional[str]) -> Dict[str, Any]:
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    return {
        "generated_at_utc": _date.datetime.now(_date.timezone.utc).isoformat(),
        "files": [describe_file(p, paper_ref, category) for p in files],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate provenance.json for HDF5 result files")
    parser.add_argument("--data_file", type=Path, action="append", required=True)
    parser.add_argument("--paper-ref", type=str, default=None, help='e.g. "Table I" or "Fig. 3"')
    parser.add_argument("--category", choices=CATEGORIES, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest = build_manifest(args.data_file, args.paper_ref, args.category)
    out = args.out or (args.data_file[0].parent / "provenance.json")
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] wrote {out}")


if __name__ == "__main__":
    main()
