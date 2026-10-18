# -*- coding: utf-8 -*-
"""算法无关契约（F06）：Trainer 只通过本接口驱动任何算法。

设计目标（对应 SRS F06/F12/F13）：MADDPG / QMIX / IQL 三种算法实现同一契约，
保证同场公平对比（换算法只改配置，不改训练器代码）。

Day08 性能扩展：新增 select_actions_batch 整队批量决策契约——
默认实现回退为逐智能体循环（行为与 select_action 完全一致），
具体算法可覆写为整队批量前向以消除单样本开销。
"""
from __future__ import annotations


class BaseAlgo:
    """多智能体算法契约基类。

    属性：
        num_agents: 智能体数量（含双方）
        obs_dim: 单智能体观测维度
        action_dim: 离散动作数
        config: 完整实验配置（YAML 加载）
    """

    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, config: dict) -> None:
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

    def select_action(self, agent_id: str, obs) -> int:
        """单智能体决策：观测 → 离散动作索引（训练期含探索）。"""
        return 0

    def select_actions_batch(self, obs: dict) -> dict:
        """整队批量决策（Day08 性能契约）：一次调用返回全部动作。

        默认实现回退为逐智能体循环（行为与 select_action 完全一致），
        具体算法可覆写为整队批量前向以消除单样本开销。
        """
        return {a: self.select_action(a, o) for a, o in obs.items()}

    def update(self, batch: list) -> dict[str, float]:
        """用一批联合转移更新网络，返回损失字典（如 {"critic_loss": ..}）。"""
        return {}

    def save(self, path: str) -> None:
        """保存算法状态（Day05 实现网络参数落盘）。"""

    def load(self, path: str) -> None:
        """恢复算法状态（Day05 实现）。"""