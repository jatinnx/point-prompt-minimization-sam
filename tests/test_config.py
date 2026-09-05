"""Config: path resolution and where SAM 3 is loaded from.

The weights live in the project at SAM-modals/sam3 so a run does not depend on
the Hugging Face cache. That default has to hold on this machine, fall back to
the hub id on a clone that has not fetched them, and be overridable without
editing the file -- and an explicit argument has to beat all three, or
``Config(sam3_model_id=...)`` would silently not take effect.
"""
from __future__ import annotations

from pathlib import Path

from pointmin.config import (PROJECT_ROOT, SAM3_HUB_ID, Config, resolve,
                             sam3_source)


def test_the_in_project_copy_is_the_default(monkeypatch, tmp_path):
    monkeypatch.delenv("POINTMIN_SAM3_PATH", raising=False)
    local = tmp_path / "sam3"
    local.mkdir()
    (local / "config.json").write_text("{}")
    assert sam3_source(local_dir=local) == str(local)


def test_a_missing_copy_falls_back_to_the_gated_hub_id(monkeypatch, tmp_path):
    """An empty or half-finished directory must not be treated as the model:
    transformers would fail deep inside from_pretrained instead of here."""
    monkeypatch.delenv("POINTMIN_SAM3_PATH", raising=False)
    (tmp_path / "sam3").mkdir()                  # exists, but no config.json
    assert sam3_source(local_dir=tmp_path / "sam3") == SAM3_HUB_ID
    assert sam3_source(local_dir=tmp_path / "absent") == SAM3_HUB_ID


def test_env_override_wins_over_both(monkeypatch, tmp_path):
    local = tmp_path / "sam3"
    local.mkdir()
    (local / "config.json").write_text("{}")
    monkeypatch.setenv("POINTMIN_SAM3_PATH", "/models/sam3")
    assert sam3_source(local_dir=local) == "/models/sam3"
    assert Config().sam3_model_id == "/models/sam3"


def test_explicit_argument_still_beats_the_environment(monkeypatch):
    monkeypatch.setenv("POINTMIN_SAM3_PATH", "/from/env")
    assert Config(sam3_model_id="/explicit").sam3_model_id == "/explicit"


def test_this_checkout_resolves_to_a_real_directory_or_the_hub(monkeypatch):
    """Whatever is on disk here, the default is something from_pretrained accepts."""
    monkeypatch.delenv("POINTMIN_SAM3_PATH", raising=False)
    source = Config().sam3_model_id
    assert source == SAM3_HUB_ID or (Path(source) / "config.json").is_file()


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
