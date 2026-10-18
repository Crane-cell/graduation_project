# -*- coding: utf-8 -*-
"""规则基线对手（任务书③）：HeuristicOpponent——纯观测函数，无内部状态、无学习。

策略：己方距球最近者朝球移动；控球者朝向对准进攻门则踢（动作 8），
否则先对准球门方向移动；其余球员同样朝球移动。
作为红队（对抗压力来源），同时构成 F13 评估的"规则基线队伍"。
"""
from __future__ import annotations

import numpy as np

from ..utils.constants import (
    ACTION_KICK,
    DIRECTIONS_8,
    OBS_ATTACK_GOAL_REL,
    OBS_BALL_REL,
    OBS_CONTROL_SELF,
    OBS_HEADING,
)
from .base import BaseAlgo


def _move_action_toward(vec: np.ndarray) -> int:
    """把目标方向映射到与其点积最大的八方向移动动作（0—7）。"""
    v = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return 9  # 无目标方向：原地不动
    v = v / norm
    return int(np.argmax(DIRECTIONS_8 @ v))


class HeuristicOpponent(BaseAlgo):
    """规则策略：实现 BaseAlgo 契约的 update 恒返回空字典（不学习）。"""

    def select_action(self, agent_id: str, obs) -> int:
        o = np.asarray(obs, dtype=np.float32)
        heading = o[OBS_HEADING : OBS_HEADING + 2]
        ball_rel = o[OBS_BALL_REL : OBS_BALL_REL + 2]
        goal_rel = o[OBS_ATTACK_GOAL_REL : OBS_ATTACK_GOAL_REL + 2]

        if o[OBS_CONTROL_SELF] > 0.5:
            # 控球：朝向已大致对准进攻门（cos 相似度阈值 0.85）→ 踢；否则先对准
            g = goal_rel / (np.linalg.norm(goal_rel) + 1e-8)
            if float(np.dot(heading, g)) >= 0.85:
                return ACTION_KICK
            return _move_action_toward(goal_rel)
        return _move_action_toward(ball_rel)

    def update(self, batch: list) -> dict[str, float]:
        return {}
