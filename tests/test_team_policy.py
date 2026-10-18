# -*- coding: utf-8 -*-
"""TeamPolicy 与规则对手测试（Day05 组合层）：分发、过滤、启发式行为。"""
import numpy as np

from src.algorithm.base import BaseAlgo
from src.algorithm.heuristic import HeuristicOpponent
from src.algorithm.replay_buffer import Transition
from src.training.team_policy import TeamPolicy
from src.utils.constants import (
    OBS_ATTACK_GOAL_REL,
    OBS_BALL_REL,
    OBS_CONTROL_SELF,
    OBS_HEADING,
)


class FakeLearner(BaseAlgo):
    def __init__(self):
        super().__init__(num_agents=3, obs_dim=32, action_dim=10, config={})
        self.seen_batches: list[list[str]] = []

    def select_action(self, agent_id, obs):
        return 42

    def update(self, batch):
        self.seen_batches.append(sorted(batch[0].obs.keys()))
        return {"loss": 1.0}


class FakeHeuristic(BaseAlgo):
    def __init__(self):
        super().__init__(num_agents=6, obs_dim=32, action_dim=10, config={})

    def select_action(self, agent_id, obs):
        return 7


def make_mixed_transition() -> Transition:
    return Transition(
        obs={"blue_0": 0, "blue_1": 1, "blue_2": 2, "red_0": 3, "red_1": 4, "red_2": 5},
        actions={k: 1 for k in ["blue_0", "blue_1", "blue_2", "red_0", "red_1", "red_2"]},
        rewards={k: 0.5 for k in ["blue_0", "blue_1", "blue_2", "red_0", "red_1", "red_2"]},
        next_obs={k: 9 for k in ["blue_0", "blue_1", "blue_2", "red_0", "red_1", "red_2"]},
        terminated={"blue_0": False, "blue_1": False, "blue_2": False, "red_0": False, "red_1": False, "red_2": False},
    )


class TestDispatch:
    def test_blue_goes_to_learner_red_to_heuristic(self):
        policy = TeamPolicy(learner=FakeLearner(), heuristic=FakeHeuristic())
        assert policy.select_action("blue_1", np.zeros(32)) == 42
        assert policy.select_action("red_0", np.zeros(32)) == 7


class TestUpdateFiltering:
    def test_update_passes_only_blue_transitions(self):
        learner = FakeLearner()
        policy = TeamPolicy(learner=learner, heuristic=FakeHeuristic())
        policy.update([make_mixed_transition(), make_mixed_transition()])
        # 一次 update = 一次 learner.update（batch 内含 2 条已过滤转移）
        assert learner.seen_batches == [["blue_0", "blue_1", "blue_2"]]

    def test_empty_batch_returns_empty(self):
        policy = TeamPolicy(learner=FakeLearner(), heuristic=FakeHeuristic())
        assert policy.update([]) == {}


def make_obs(overrides: dict | None = None) -> np.ndarray:
    """构造 32 维测试观测（键为观测偏移常量，可为整数）。"""
    o = np.zeros(32, dtype=np.float32)
    for key, val in (overrides or {}).items():
        o[key] = val
    return o


class TestHeuristic:
    def test_kicks_when_controlled_and_aligned(self, algo_cfg):
        """控球 + 朝向对准进攻门（夹角小）→ 踢（动作 8）。"""
        opp = HeuristicOpponent(6, 32, 10, algo_cfg)
        o = make_obs({OBS_CONTROL_SELF: 1.0})
        o[OBS_HEADING : OBS_HEADING + 2] = [1.0, 0.0]
        o[OBS_ATTACK_GOAL_REL : OBS_ATTACK_GOAL_REL + 2] = [10.0, 0.0]
        assert opp.select_action("red_0", o) == 8

    def test_aligns_first_when_not_facing_goal(self, algo_cfg):
        opp = HeuristicOpponent(6, 32, 10, algo_cfg)
        o = make_obs({OBS_CONTROL_SELF: 1.0})
        o[OBS_HEADING : OBS_HEADING + 2] = [0.0, 1.0]     # 朝北
        o[OBS_ATTACK_GOAL_REL : OBS_ATTACK_GOAL_REL + 2] = [10.0, 0.0]  # 门在东
        assert opp.select_action("red_0", o) == 0         # 朝东移动对准

    def test_moves_toward_ball_when_not_controlling(self, algo_cfg):
        opp = HeuristicOpponent(6, 32, 10, algo_cfg)
        o = make_obs()
        o[OBS_BALL_REL : OBS_BALL_REL + 2] = [5.0, 0.0]
        assert opp.select_action("red_1", o) == 0

    def test_stays_when_no_direction(self, algo_cfg):
        opp = HeuristicOpponent(6, 32, 10, algo_cfg)
        o = make_obs()
        o[OBS_BALL_REL : OBS_BALL_REL + 2] = [0.0, 0.0]
        assert opp.select_action("red_2", o) == 9
