"""Config: path resolution and the checkpoint override a clone needs.

SAM1_CHECKPOINT_CANDIDATES are absolute paths on the machine this was built on.
Without an override, a clone anywhere else cannot point at its own copy.
"""
from __future__ import annotations

from pathlib import Path

from pointmin.config import PROJECT_ROOT, Config, resolve


def test_checkpoint_override_wins(monkeypatch, tmp_path):
    ckpt = tmp_path / "sam_vit_b.pth"
    ckpt.write_bytes(b"not really a checkpoint")
    monkeypatch.setenv("POINTMIN_SAM1_CHECKPOINT", str(ckpt))
    assert Config().sam1_checkpoint == str(ckpt)


def test_override_is_honoured_even_when_it_does_not_exist(monkeypatch):
    """A wrong path must surface as 'checkpoint not found: <your path>', not as a
    silent fall back to a path the user did not ask for."""
    monkeypatch.setenv("POINTMIN_SAM1_CHECKPOINT", "/nope/sam.pth")
    assert Config().sam1_checkpoint == "/nope/sam.pth"


def test_explicit_argument_still_beats_the_environment(monkeypatch):
    monkeypatch.setenv("POINTMIN_SAM1_CHECKPOINT", "/from/env.pth")
    assert Config(sam1_checkpoint="/explicit.pth").sam1_checkpoint == "/explicit.pth"


def test_relative_paths_anchor_to_the_project_not_the_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert resolve("pointmin") == PROJECT_ROOT / "pointmin"
    assert Config().images == PROJECT_ROOT / "data/dlrsd/images"


def test_absolute_paths_pass_through():
    assert resolve("/tmp/x") == Path("/tmp/x")


def test_artifacts_dir_is_created_on_access(tmp_path):
    target = tmp_path / "nested" / "artifacts"
    assert not target.exists()
    assert Config(artifacts_dir=str(target)).artifacts == target
    assert target.is_dir()
