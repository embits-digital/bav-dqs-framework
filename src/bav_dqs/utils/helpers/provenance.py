"""Provenance helpers: capture which code and which config produced a result file."""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def sha256_file(path: Path | str) -> str:
    """SHA-256 of a file's bytes (hex digest)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def git_info(start: Optional[Path | str] = None) -> Dict[str, Any]:
    """Current Git commit and dirty flag for the repo containing ``start`` (default: this package).

    Returns ``"unknown"`` values when Git or the repository is unavailable,
    so provenance capture never breaks a run.
    """
    cwd = Path(start) if start is not None else Path(__file__).resolve().parent
    if cwd.is_file():
        cwd = cwd.parent
    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"git_commit": "unknown", "git_dirty": "unknown"}
    # Tracked-file changes only: untracked outputs (e.g. results/*.h5) must not flag every run as dirty.
    status = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    dirty = "unknown" if status is None else bool(status)
    return {"git_commit": commit, "git_dirty": dirty}


def collect_provenance(config_path: Path | str) -> Dict[str, Any]:
    """Metadata block to embed in the HDF5 root attributes of every run."""
    config_path = Path(config_path)
    info: Dict[str, Any] = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path) if config_path.exists() else "unknown",
    }
    info.update(git_info())
    return info
