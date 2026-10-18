# -*- coding: utf-8 -*-
"""Day07 数据驱动与集成测试：配置校验/兜底/告警 + 真实整链路训练与断点续训。

覆盖教案 Day07 验收面：
  - 配置：缺字段兜底、未知字段告警、类型错/范围错/段类型错报 ValueError
  - 默认 YAML 全量校验通过（无告警，含消融开关）
  - 整链路：真实配置 + 真实 MADDPG + 回放缓冲 + 训练器 + 检查点 → 断点恢复续训
"""
from pathlib import Path

import pytest

from src.env.soccer_env import SoccerEnv
from src.utils.checkpoint import CheckpointManager, SCHEMA_VERSION
from src.utils.config import DEFAULTS, load_config, validate_config


class TestConfigValidation:
    def test_missing_fields_filled_with_defaults(self):
        """缺字段兜底：只给两个字段，其余全部取默认值且类型正确。"""
        merged = validate_config(
            {"algorithm": {"name": "qmix", "lr": 5e-4}, "train": {"total_steps": 100}}
        )
        assert merged["algorithm"]["name"] == "qmix"
        assert merged["algorithm"]["lr"] == 5e-4
        assert merged["algorithm"]["gamma"] == pytest.approx(DEFAULTS["algorithm"]["gamma"])
        assert merged["algorithm"]["ablation"]["value_decomposition"] is True
        assert merged["env"]["num_per_side"] == 3
        assert merged["env"]["field"]["width"] == pytest.approx(20.0)
        assert merged["train"]["total_steps"] == 100
        assert merged["eval"]["n_episodes"] == 1000
        assert merged["logging"]["log_dir"] == "runs"

    def test_unknown_fields_warn_and_ignored(self):
        """未知字段（含顶层与子段）：UserWarning 提示，但配置正常返回。"""
        with pytest.warns(UserWarning) as rec:
            merged = validate_config(
                {"algorithm": {"learnr": 0.1}, "extra_section": {"a": 1}, "env": {"unknown_key": 2}}
            )
        msg = "\n".join(str(w.message) for w in rec)
        assert "learnr" in msg and "extra_section" in msg and "unknown_key" in msg
        assert "learnr" not in merged["algorithm"]
        assert "extra_section" not in merged

    def test_type_error_raises_valueerror(self):
        """类型违例（字符串当整数）：显式 ValueError 而非静默。"""
        with pytest.raises(ValueError, match="batch_size"):
            validate_config({"algorithm": {"batch_size": "32"}})

    def test_bool_field_rejects_non_bool(self):
        """布尔字段只收 bool（或 0/1），拒收字符串。"""
        with pytest.raises(ValueError, match="value_decomposition"):
            validate_config({"algorithm": {"ablation": {"value_decomposition": "yes"}}})
        assert validate_config({"algorithm": {"ablation": {"communication": 1}}})["algorithm"][
            "ablation"
        ]["communication"] is True

    def test_range_error_raises_valueerror(self):
        """数值越界（epsilon>1）：显式 ValueError 并点名字段。"""
        with pytest.raises(ValueError, match="epsilon"):
            validate_config({"algorithm": {"epsilon": 1.5}})

    def test_enum_error_raises_valueerror(self):
        """枚举违例（算法名不在允许集合）：显式 ValueError。"""
        with pytest.raises(ValueError, match="name"):
            validate_config({"algorithm": {"name": "ddpg"}})

    def test_section_type_error_raises_valueerror(self):
        """段类型违例（algorithm 不是字典）：显式 ValueError。"""
        with pytest.raises(ValueError, match="algorithm"):
            validate_config({"algorithm": "maddpg"})

    def test_int_field_accepts_integral_float(self):
        """整数域兼容整值浮点（YAML 里 3.0 等价 3）。"""
        merged = validate_config({"env": {"num_per_side": 3.0}})
        assert merged["env"]["num_per_side"] == 3
        assert isinstance(merged["env"]["num_per_side"], int)


class TestDefaultYaml:
    def test_default_yaml_validates_without_warning(self):
        """默认 YAML：全量校验通过且零告警，消融开关完整。"""
        root = Path(__file__).resolve().parents[1]
        import warnings

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cfg = load_config(root / "src" / "configs" / "default.yaml")
        assert not rec
        assert cfg["env"]["obs_dim"] == 32
        assert cfg["algorithm"]["name"] in {"maddpg", "qmix", "iql"}
        assert cfg["algorithm"]["ablation"] == {
            "communication": False,
            "role_assignment": True,
            "value_decomposition": True,
        }


class TestFullPipeline:
    def test_train_and_resume_from_checkpoint(self, env_cfg, algo_cfg, tmp_path):
        """整链路集成：真实配置 + 真实 MADDPG + 缓冲 + 训练器，8 步存档 → 断点续训至 16。"""
        from src.algorithm.maddpg import MADDPG
        from src.algorithm.replay_buffer import ReplayBuffer
        from src.training.trainer import Trainer
        from src.utils.logger import MetricLogger

        cfg = dict(env_cfg)
        cfg["algorithm"] = {**algo_cfg["algorithm"], "name": "maddpg"}
        cfg["train"] = {"total_steps": 8, "eval_interval": 100, "checkpoint_interval": 4}

        run_dir = tmp_path / "runs"
        logger = MetricLogger(run_dir)
        ckpt = CheckpointManager(run_dir / "ckpt")
        buffer = ReplayBuffer(100)

        # 第一段：0 → 8
        env1 = SoccerEnv(cfg)
        algo1 = MADDPG(num_agents=6, obs_dim=32, action_dim=10, config=cfg)
        r1 = Trainer(env1, algo1, buffer, logger, cfg, ckpt).train()
        assert r1["global_step"] == 8
        assert r1["buffer_size"] == 8

        # 检查点内容：版本号 + 步数元信息
        state = ckpt.load("latest")
        assert state["schema_version"] == SCHEMA_VERSION == 1
        assert state["global_step"] == 8
        assert "saved_at" in state

        # 第二段：从中断点 8 → 16（total_steps 为绝对预算；续训需传回原始总预算）
        cfg2 = dict(cfg)
        cfg2["train"] = {**cfg["train"], "total_steps": 16}
        env2 = SoccerEnv(cfg2)
        algo2 = MADDPG(num_agents=6, obs_dim=32, action_dim=10, config=cfg2)
        trainer2 = Trainer(env2, algo2, buffer, logger, cfg2, ckpt)
        trainer2._global_step = state["global_step"]
        r2 = trainer2.train()
        assert r2["global_step"] == 16
        assert ckpt.load("latest")["global_step"] == 16
        logger.close()

    def test_legacy_checkpoint_loads_with_warning(self, tmp_path):
        """旧格式检查点（无 schema_version）：兼容读取并告警，字段补齐。"""
        import torch

        ckpt = CheckpointManager(tmp_path / "ckpt")
        path = tmp_path / "ckpt" / "legacy.pt"
        torch.save({"global_step": 42, "extra": [1, 2]}, path)

        with pytest.warns(UserWarning, match="旧格式"):
            state = ckpt.load("legacy")
        assert state["global_step"] == 42
        assert state["schema_version"] == SCHEMA_VERSION
        assert state["saved_at"] == 0.0