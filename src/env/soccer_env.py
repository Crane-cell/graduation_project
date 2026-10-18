# -*- coding: utf-8 -*-
"""自研轻量 2D 足球仿真环境（F01，Q7）——PettingZoo Parallel API 兼容。

Day05 真实现：物理（移动/踢球/碰撞/进球）、观测编码（32 维，LLD 第 3 章）、
奖励计算（团队稀疏 + 个体塑形，LLD 第 4 章）。设计约束（HLD 红线）：
本层可以依赖 gymnasium/pettingzoo；算法层禁止 import 本层。

可复现性（NF-02）：全部随机性只经 reset(seed) 建立的 numpy Generator；
给定相同种子与相同动作序列，轨迹逐元素一致（Day06 断言）。
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from ..agent.roles import Role, role_onehot
from ..utils.constants import (
    ACTION_KICK,
    ACTION_STAY,
    DIRECTIONS_8,
    MOVE_ACTION_COUNT,
    OBS_ATTACK_GOAL_REL,
    OBS_BALL_POS,
    OBS_BALL_REL,
    OBS_BALL_VEL,
    OBS_CONTROL_SELF,
    OBS_CONTROL_TEAM,
    OBS_DEFEND_GOAL_REL,
    OBS_HEADING,
    OBS_OPPONENT_REL,
    OBS_ROLE_ONEHOT,
    OBS_SELF_POS,
    OBS_TEAMMATE_REL,
    OBS_TEAM_ID,
    OBS_DIM,
)


class SoccerEnv(ParallelEnv):
    """3v3 起步的双方对抗足球环境（blue_0.. 攻右门 / red_0.. 攻左门）。

    坐标系：x ∈ [0, width]（进攻方向蓝→+x），y ∈ [0, height]。
    """

    metadata = {"name": "soccer_marl_v0", "is_parallelizable": True}

    def __init__(self, config: dict) -> None:
        env_cfg = config["env"]
        field_cfg = env_cfg.get("field", {})
        rew_cfg = env_cfg.get("reward", {})

        self.n_per_side = int(env_cfg.get("num_per_side", 3))
        self.obs_dim = int(env_cfg.get("obs_dim", OBS_DIM))
        self.action_dim = int(env_cfg.get("action_dim", 10))
        self.max_episode_steps = int(env_cfg.get("max_episode_steps", 500))
        self.width = float(field_cfg.get("width", 20.0))
        self.height = float(field_cfg.get("height", 15.0))
        self.goal_half_width = float(env_cfg.get("goal_half_width", 2.0))

        # 物理常量（LLD 第 4 章；模块级常量而非字面散值）
        self.agent_radius = 0.3
        self.ball_radius = 0.15
        self.move_speed = 1.0
        self.kick_speed = 3.0
        self.ball_friction = 0.9
        self.control_radius = 0.5
        self.ball_inherit = 0.3  # 碰撞时球继承智能体移动速度的比例

        # 奖励（LLD 第 4 章；权重全部外置可配）
        self.goal_reward = float(rew_cfg.get("goal", 10.0))
        self.control_bonus = float(rew_cfg.get("control", 0.02))
        self.advance_reward = float(rew_cfg.get("advance", 0.5))
        self.approach_reward = float(rew_cfg.get("approach", 0.005))

        self.possible_agents: list[str] = (
            [f"blue_{i}" for i in range(self.n_per_side)] + [f"red_{i}" for i in range(self.n_per_side)]
        )
        self.agents: list[str] = []
        self._rng: np.random.Generator | None = None

        total = 2 * self.n_per_side
        self._pos = np.zeros((total, 2), dtype=np.float64)
        self._heading = np.zeros((total, 2), dtype=np.float64)
        self._ball = np.zeros(2, dtype=np.float64)
        self._ball_vel = np.zeros(2, dtype=np.float64)
        self._step_count = 0

        self._obs_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self._act_space = spaces.Discrete(self.action_dim)

    # ---------- 内部工具 ----------

    def _is_blue(self, idx: int) -> bool:
        return idx < self.n_per_side

    def _teammates_of(self, idx: int) -> list[int]:
        base = range(0, self.n_per_side) if self._is_blue(idx) else range(self.n_per_side, 2 * self.n_per_side)
        return [j for j in base if j != idx]

    def _opponents_of(self, idx: int) -> list[int]:
        if self._is_blue(idx):
            return list(range(self.n_per_side, 2 * self.n_per_side))
        return list(range(0, self.n_per_side))

    def _role_of(self, idx: int) -> Role:
        in_team = idx if self._is_blue(idx) else idx - self.n_per_side
        return Role(in_team % 3)

    def _attack_goal(self, idx: int) -> np.ndarray:
        """进攻门中心：蓝队攻右门，红队攻左门。"""
        if self._is_blue(idx):
            return np.array([self.width, self.height / 2.0])
        return np.array([0.0, self.height / 2.0])

    def _defend_goal(self, idx: int) -> np.ndarray:
        if self._is_blue(idx):
            return np.array([0.0, self.height / 2.0])
        return np.array([self.width, self.height / 2.0])

    def _observe(self, idx: int) -> np.ndarray:
        """观测编码（LLD 第 3 章布局）：每次新建数组（不返回内部引用）。

        归一化契约（SRS GWT-2）：全部元素 ∈ [-1,1]——自身/球位置以场地中心归一，
        相对位置按整场宽高归一，最后整体裁剪兜底。
        """
        o = np.zeros(self.obs_dim, dtype=np.float32)
        center = np.array([self.width / 2.0, self.height / 2.0])
        field = np.array([self.width, self.height])

        p = self._pos[idx]
        o[OBS_SELF_POS : OBS_SELF_POS + 2] = (p - center) / center
        o[OBS_HEADING : OBS_HEADING + 2] = self._heading[idx]
        o[OBS_BALL_POS : OBS_BALL_POS + 2] = (self._ball - center) / center
        o[OBS_BALL_REL : OBS_BALL_REL + 2] = (self._ball - p) / field
        o[OBS_BALL_VEL : OBS_BALL_VEL + 2] = self._ball_vel / self.kick_speed

        for k, j in enumerate(self._teammates_of(idx)):
            o[OBS_TEAMMATE_REL + 2 * k : OBS_TEAMMATE_REL + 2 * k + 2] = (self._pos[j] - p) / field
        for k, j in enumerate(self._opponents_of(idx)):
            o[OBS_OPPONENT_REL + 2 * k : OBS_OPPONENT_REL + 2 * k + 2] = (self._pos[j] - p) / field

        o[OBS_ROLE_ONEHOT : OBS_ROLE_ONEHOT + 3] = role_onehot(self._role_of(idx))
        o[OBS_TEAM_ID] = 1.0 if self._is_blue(idx) else -1.0

        dist_ball = float(np.linalg.norm(self._ball - p))
        o[OBS_CONTROL_SELF] = 1.0 if dist_ball < self.control_radius else 0.0
        team_control = any(
            np.linalg.norm(self._ball - self._pos[j]) < self.control_radius for j in self._teammates_of(idx)
        )
        o[OBS_CONTROL_TEAM] = 1.0 if (o[OBS_CONTROL_SELF] > 0 or team_control) else 0.0

        o[OBS_ATTACK_GOAL_REL : OBS_ATTACK_GOAL_REL + 2] = (self._attack_goal(idx) - p) / field
        o[OBS_DEFEND_GOAL_REL : OBS_DEFEND_GOAL_REL + 2] = (self._defend_goal(idx) - p) / field
        return np.clip(o, -1.0, 1.0)

    def _observe_all(self) -> dict[str, np.ndarray]:
        return {a: self._observe(i) for i, a in enumerate(self.agents)}

    # ---------- PettingZoo ParallelEnv 契约 ----------

    def reset(self, seed: int | None = None, options: dict | None = None):
        """重置一局：随机站位（蓝左红右），球置中心。"""
        self._rng = np.random.default_rng(seed)
        n = self.n_per_side
        self._step_count = 0
        for i in range(n):
            self._pos[i] = [self._rng.uniform(2.0, 6.0), self._rng.uniform(1.0, self.height - 1.0)]
            self._pos[i + n] = [self._rng.uniform(self.width - 6.0, self.width - 2.0), self._rng.uniform(1.0, self.height - 1.0)]
        self._heading[:n] = [1.0, 0.0]
        self._heading[n:] = [-1.0, 0.0]
        self._ball = np.array([self.width / 2.0, self.height / 2.0])
        self._ball_vel = np.zeros(2)
        self.agents = list(self.possible_agents)
        return self._observe_all(), {a: {} for a in self.agents}

    def step(self, actions: dict[str, int]):
        if not self.agents:
            raise RuntimeError("对局已结束：请先调用 reset()。")

        n = self.n_per_side
        total = 2 * n
        prev_ball = self._ball.copy()
        prev_dist = np.linalg.norm(self._pos - self._ball, axis=1)

        # --- 1. 动作解码与踢球 ---
        move_vec = np.zeros((total, 2))
        for idx, aid in enumerate(self.possible_agents):
            act = actions.get(aid)
            if not isinstance(act, (int, np.integer)) or act < 0 or act >= self.action_dim:
                raise ValueError(f"非法动作 {act!r}（来自 {aid}，合法范围 [0, {self.action_dim})）")
            if act < MOVE_ACTION_COUNT:
                d = DIRECTIONS_8[int(act)]
                move_vec[idx] = d * self.move_speed
                self._heading[idx] = d
            elif act == ACTION_KICK:
                if np.linalg.norm(self._ball - self._pos[idx]) < self.control_radius:
                    self._ball_vel = self._heading[idx] * self.kick_speed
            elif act == ACTION_STAY:
                pass

        # --- 2. 位移 + 边界 ---
        self._pos = self._pos + move_vec
        self._pos[:, 0] = np.clip(self._pos[:, 0], self.agent_radius, self.width - self.agent_radius)
        self._pos[:, 1] = np.clip(self._pos[:, 1], self.agent_radius, self.height - self.agent_radius)

        # --- 3. 智能体间碰撞：沿连线推开 ---
        for i in range(total):
            for j in range(i + 1, total):
                diff = self._pos[i] - self._pos[j]
                dist = float(np.linalg.norm(diff))
                min_d = 2.0 * self.agent_radius
                if dist < min_d and dist > 1e-9:
                    push = (min_d - dist) / 2.0
                    diff = diff / dist
                    self._pos[i] += diff * push
                    self._pos[j] -= diff * push

        # --- 4. 球物理 + 球-智能体碰撞 ---
        self._ball = self._ball + self._ball_vel
        self._ball_vel = self._ball_vel * self.ball_friction
        for idx in range(total):
            diff = self._ball - self._pos[idx]
            dist = float(np.linalg.norm(diff))
            min_d = self.ball_radius + self.agent_radius
            if dist < min_d and dist > 1e-9:
                diff = diff / dist
                self._ball = self._pos[idx] + diff * min_d
                self._ball_vel = self._ball_vel + move_vec[idx] * self.ball_inherit

        # --- 5. 进球与围墙（先判门再贴墙）---
        goal = 0  # 0 无 / 1 蓝方进球 / 2 红方进球
        cy = self.height / 2.0
        in_goal_y = abs(self._ball[1] - cy) <= self.goal_half_width
        if self._ball[0] <= 0.0 and in_goal_y:
            goal = 2  # 球进左门 → 红方得分
        elif self._ball[0] >= self.width and in_goal_y:
            goal = 1  # 球进右门 → 蓝方得分
        else:
            if self._ball[0] < self.ball_radius:
                self._ball[0] = self.ball_radius
                self._ball_vel[0] = abs(self._ball_vel[0]) * 0.5
            elif self._ball[0] > self.width - self.ball_radius:
                self._ball[0] = self.width - self.ball_radius
                self._ball_vel[0] = -abs(self._ball_vel[0]) * 0.5
        if self._ball[1] < self.ball_radius:
            self._ball[1] = self.ball_radius
            self._ball_vel[1] = abs(self._ball_vel[1]) * 0.5
        elif self._ball[1] > self.height - self.ball_radius:
            self._ball[1] = self.height - self.ball_radius
            self._ball_vel[1] = -abs(self._ball_vel[1]) * 0.5

        # --- 6. 奖励（LLD 第 4 章）---
        rew = np.zeros(total)
        if goal == 1:
            rew[:n] += self.goal_reward
            rew[n:] -= self.goal_reward
        elif goal == 2:
            rew[n:] += self.goal_reward
            rew[:n] -= self.goal_reward
        dx = float(self._ball[0] - prev_ball[0])
        rew[:n] += self.advance_reward * dx          # 蓝攻 +x
        rew[n:] += self.advance_reward * (-dx)       # 红攻 -x
        new_dist = np.linalg.norm(self._pos - self._ball, axis=1)
        rew += self.approach_reward * np.maximum(prev_dist - new_dist, 0.0)
        rew[new_dist < self.control_radius] += self.control_bonus

        # --- 7. 终局 ---
        self._step_count += 1
        terminated = goal != 0 or self._step_count >= self.max_episode_steps
        obs = self._observe_all()
        rewards = {a: float(rew[i]) for i, a in enumerate(self.possible_agents)}
        terminations = {a: terminated for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        infos = {a: {"goal": goal, "step": self._step_count} for a in self.possible_agents}
        if terminated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._act_space
