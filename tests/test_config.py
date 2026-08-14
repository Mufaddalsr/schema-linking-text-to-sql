"""Tests for src/schema_linking/utils/config.py."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from textwrap import dedent

import pytest

from schema_linking.utils.config import (
    Config,
    DataConfig,
    DEFAULT_CONFIG_PATH,
    OutputsConfig,
    load_config,
)


def _all_path_fields(cfg: Config) -> list[Path]:
    paths: list[Path] = []
    for section in (cfg.data, cfg.outputs):
        for f in fields(section):
            paths.append(getattr(section, f.name))
    return paths


def test_returns_config_with_nested_sections() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.outputs, OutputsConfig)


def test_all_paths_are_Path_objects() -> None:
    cfg = load_config()
    for p in _all_path_fields(cfg):
        assert isinstance(p, Path), f"expected Path, got {type(p).__name__}"


def test_all_paths_are_absolute() -> None:
    cfg = load_config()
    for p in _all_path_fields(cfg):
        assert p.is_absolute(), f"path is not absolute: {p}"


def test_data_paths_exist_on_disk() -> None:
    cfg = load_config()
    assert cfg.data.spider_dir.is_dir(), cfg.data.spider_dir
    assert cfg.data.processed_dir.is_dir(), cfg.data.processed_dir
    assert cfg.data.taniguchi_splits_dir.is_dir(), cfg.data.taniguchi_splits_dir


def test_output_paths_creatable() -> None:
    cfg = load_config()
    for p in (
        cfg.outputs.predictions_dir,
        cfg.outputs.results_dir,
        cfg.outputs.logs_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)
        assert p.is_dir(), f"could not create or find {p}"


def test_default_config_path_resolves_from_module_not_cwd(tmp_path, monkeypatch) -> None:
    """load_config() with no args must work regardless of CWD."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.data.spider_dir.is_dir()


def test_load_with_explicit_path_resolves_relative_to_yaml(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "data" / "x").mkdir(parents=True)
    yaml_path = sub / "config.yaml"
    yaml_path.write_text(
        dedent(
            """
            data:
              spider_dir: data/x
              processed_dir: data/x
              taniguchi_splits_dir: data/x
            outputs:
              predictions_dir: out/p
              results_dir: out/r
              logs_dir: out/l
            """
        ).strip()
    )
    cfg = load_config(yaml_path)
    assert cfg.data.spider_dir == (sub / "data" / "x").resolve()
    assert cfg.outputs.logs_dir == (sub / "out" / "l").resolve()


def test_missing_file_raises_FileNotFoundError(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_missing_section_raises_KeyError(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("data:\n  spider_dir: x\n  processed_dir: y\n")
    with pytest.raises(KeyError):
        load_config(yaml_path)


def test_default_path_is_repo_root_config_yaml() -> None:
    assert DEFAULT_CONFIG_PATH.name == "config.yaml"
    assert DEFAULT_CONFIG_PATH.is_file()
