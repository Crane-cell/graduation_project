# -*- coding: utf-8 -*-
"""工具层测试（F05/F10/NF-02）：配置加载与检查点存取。"""
import pytest
import yaml

from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config
from src.utils.seed import set_global_seed


class TestConfig:
    def test_load_real_default_yaml(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "src" / "configs" / "default.yaml")
        assert cfg["env"]["obs_dim"] == 32
        assert cfg["algorithm"]["name"] in {"maddpg", "qmix", "iql"}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("Z:/not/exist.yaml")

    def test_non_mapping_top_level_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(p)


class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        mgr.save("latest", {"global_step": 42, "extra": [1, 2]})
        state = mgr.load("latest")
        assert state["global_step"] == 42
        assert state["extra"] == [1, 2]
        assert "saved_at" in state  # 元信息字段

    def test_latest_returns_none_when_empty(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "empty")
        assert mgr.latest() is None

    def test_load_missing_raises(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt2")
        with pytest.raises(FileNotFoundError):
            mgr.load("ghost")


class TestSeed:
    def test_set_global_seed_reproducible(self):
        import numpy as np

        set_global_seed(7)
        a = np.random.rand(4)
        set_global_seed(7)
        b = np.random.rand(4)
        assert (a == b).all()
