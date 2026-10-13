import hashlib
import shutil
import subprocess

import pytest

from bav_dqs.utils.helpers.provenance import collect_provenance, git_info, sha256_file

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def test_sha256_file(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_bytes(b"a: 1\n")
    assert sha256_file(p) == hashlib.sha256(b"a: 1\n").hexdigest()


def test_collect_provenance_missing_config_does_not_raise(tmp_path):
    info = collect_provenance(tmp_path / "missing.yaml")
    assert info["config_sha256"] == "unknown"
    assert {"config_path", "git_commit", "git_dirty"} <= set(info)


def test_git_info_outside_repo_is_unknown(tmp_path):
    # tmp_path is normally outside any repository; if the environment nests it in one, skip.
    info = git_info(tmp_path)
    if info["git_commit"] != "unknown":
        pytest.skip("tmp_path is inside a git repository in this environment")
    assert info == {"git_commit": "unknown", "git_dirty": "unknown"}


@needs_git
def test_git_info_commit_and_dirty_flag(tmp_path):
    def run(cmd):
        subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True)

    cfg = tmp_path / "c.yaml"
    cfg.write_text("a: 1\n")
    run(["git", "init", "-q"])
    run(["git", "config", "user.email", "t@example.com"])
    run(["git", "config", "user.name", "t"])
    run(["git", "add", "c.yaml"])
    run(["git", "commit", "-qm", "init"])
    (tmp_path / "untracked_result.h5").write_bytes(b"x")  # untracked outputs must not mark dirty

    info = git_info(tmp_path)
    assert len(info["git_commit"]) == 40
    assert info["git_dirty"] is False

    cfg.write_text("a: 2\n")  # tracked file modified
    assert git_info(tmp_path)["git_dirty"] is True
