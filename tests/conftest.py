# -*- coding: utf-8 -*-
"""pytest 共享配置：把项目根加入 sys.path（tests 直接 import src.*）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture()
def env_cfg() -> dict:
    """环境测试用小配置。"""
    return {
        "env": {
            "num_per_side": 3,
            "obs_dim": 32,
            "action_dim": 10,
            "max_episode_steps": 500,
            "goal_half_width": 2.0,
            "field": {"width": 20.0, "height": 15.0},
            "reward": {"goal": 10.0, "control": 0.02, "advance": 0.5, "approach": 0.005},
        }
    }


@pytest.fixture()
def algo_cfg() -> dict:
    """算法测试用小配置（小网络保测试速度；ε=0 保证决策确定性）。"""
    return {
        "algorithm": {
            "hidden_dim": 16,
            "embed_dim": 8,
            "lr": 1e-3,
            "gamma": 0.9,
            "tau_soft": 0.05,
            "epsilon": 0.0,
            "batch_size": 8,
            "buffer_size": 64,
            "update_start": 4,
            "update_every": 2,
        }
    }


@pytest.fixture()
def make_batch():
    """合成全队联合转移批（固定种子可复现）。"""

    def _make(n_agents: int = 3, obs_dim: int = 8, action_dim: int = 4,
              batch_size: int = 16, seed: int = 0) -> list:
        from src.algorithm.replay_buffer import Transition

        rng = np.random.default_rng(seed)
        ids = [f"blue_{i}" for i in range(n_agents)]
        batch = []
        for _ in range(batch_size):
            batch.append(
                Transition(
                    obs={a: rng.normal(size=obs_dim).astype(np.float32) for a in ids},
                    actions={a: int(rng.integers(action_dim)) for a in ids},
                    rewards={a: float(rng.normal()) for a in ids},
                    next_obs={a: rng.normal(size=obs_dim).astype(np.float32) for a in ids},
                    terminated={a: bool(rng.random() < 0.2) for a in ids},
                )
            )
        return batch

    return _make
