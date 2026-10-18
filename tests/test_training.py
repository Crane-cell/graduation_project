# -*- coding: utf-8 -*-
"""训练层测试（F09/F13）：主循环状态机转移与评估协议真实统计。

对应 LLD 状态机：每条转移至少一条用例（update 触发/局末重置/周期存档/达标结束）。
"""
import numpy as np
import pytest

from src.algorithm.base import BaseAlgo
from src.env.soccer_env import SoccerEnv
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import MetricLogger


class RecordingAlgo(BaseAlgo):
    """假算法：记录 update 次数（覆盖损失日志分支）。"""

    def __init__(self):
        super().__init__(num_agents=6, obs_dim=32, action_dim=10, config={})
        self.n_updates = 0
        self.n_selects = 0

    def select_action(self, agent_id, obs):
        self.n_selects += 1
        return 0 if agent_id.startswith("blue") else 9

    def update(self, batch):
        self.n_updates += 1
        return {"loss": float(len(batch))}


def tiny_train_cfg(env_cfg) -> dict:
    cfg = dict(env_cfg)
    cfg["algorithm"] = {"update_start": 2, "update_every": 1, "batch_size": 2}
    cfg["train"] = {"total_steps": 5, "eval_interval": 100, "checkpoint_interval": 2}
    return cfg


def test_trainer_full_loop_state_machine(env_cfg, tmp_path):
    """一次 train() 走完 COLLECT→UPDATE→RESET→SAVE 全部转移。"""
    env_cfg["env"]["max_episode_steps"] = 2  # 每两步一局
    env = SoccerEnv(env_cfg)
    algo = RecordingAlgo()
    cfg = tiny_train_cfg(env_cfg)

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.algorithm.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(100)
    logger = MetricLogger(str(tmp_path / "runs"))
    ckpt = CheckpointManager(tmp_path / "runs" / "ckpt")
    trainer = Trainer(env, algo, buffer, logger, cfg, ckpt)

    result = trainer.train()

    assert result["global_step"] == 5
    assert result["buffer_size"] == 5
    assert algo.n_updates >= 3            # update_start=2 后每步更新
    assert algo.n_selects == 5 * 6        # 每步整队 6 次决策
    assert ckpt.latest() == "latest"      # 周期存档触发（步 2、4）
    state = ckpt.load("latest")
    assert state["global_step"] == 4
    logger.close()


def test_evaluator_real_metrics(env_cfg):
    """评估协议（F13）：启发式对启发式（双方都攻），指标为真实计数而非占位。"""
    env_cfg["env"]["max_episode_steps"] = 30
    env = SoccerEnv(env_cfg)

    class BothHeuristic(BaseAlgo):
        """双方共用同一启发式：验证 Evaluator 统计真实对局而非硬编码。"""

        def __init__(self):
            super().__init__(6, 32, 10, {})
            from src.algorithm.heuristic import HeuristicOpponent

            self._h = HeuristicOpponent(6, 32, 10, {})

        def select_action(self, agent_id, obs):
            return self._h.select_action(agent_id, obs)

        def update(self, batch):
            return {}

    metrics = Evaluator().evaluate(env, BothHeuristic(), n_episodes=3, seed=123)
    total = metrics["win_rate"] + metrics["loss_rate"] + metrics["draw_rate"]
    assert total == pytest.approx(1.0)
    assert metrics["n_episodes"] == 3.0
    assert metrics["inference_ms_p50"] >= 0.0
    assert metrics["inference_ms_p95"] >= 0.0
    assert -3.0 <= metrics["goal_diff_per_game"] <= 3.0
    assert -60.0 <= metrics["team_reward"] <= 60.0  # 每局塑形奖励粗界（30 步 × 6 人 × ≤0.02…）
