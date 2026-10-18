# -*- coding: utf-8 -*-
"""TeamPolicy（Day05，LLD 第 5 章）：蓝队=学习智能体，红队=规则基线对手。

实现 BaseAlgo 契约的组合器：select_action 按队伍前缀分发；
update 只把蓝队的联合转移过滤后转交 learner（红队不学习）；
save/load 委托 learner。Trainer 与契约零改动（HLD 设计目标"结构可测试"）。
"""
from __future__ import annotations

from ..algorithm.replay_buffer import Transition


class TeamPolicy:
    def __init__(self, learner, heuristic, learning_prefix: str = "blue") -> None:
        self.learner = learner
        self.heuristic = heuristic
        self.learning_prefix = learning_prefix

    # ---------- 分发 ----------

    def select_action(self, agent_id: str, obs) -> int:
        if agent_id.startswith(self.learning_prefix):
            return self.learner.select_action(agent_id, obs)
        return self.heuristic.select_action(agent_id, obs)

    def select_actions_batch(self, obs: dict) -> dict:
        """整队批量决策（Day08 优化）：蓝队走 learner 批量前向，红队逐智能体。"""
        blue = {a: o for a, o in obs.items() if a.startswith(self.learning_prefix)}
        red = {a: o for a, o in obs.items() if not a.startswith(self.learning_prefix)}
        actions: dict = {}
        if blue:
            actions.update(self.learner.select_actions_batch(blue))
        if red:
            actions.update(self.heuristic.select_actions_batch(red))
        return actions

    def _blue_ids(self, batch: list) -> list[str]:
        return sorted(
            (k for k in batch[0].obs.keys() if k.startswith(self.learning_prefix)),
            key=lambda s: int(s.rsplit("_", 1)[1]),
        )

    def update(self, batch: list) -> dict[str, float]:
        """过滤出学习队的联合转移，转交 learner 更新。"""
        if not batch:
            return {}
        blue = self._blue_ids(batch)

        def pick(d: dict) -> dict:
            return {k: d[k] for k in blue}

        filtered = [
            Transition(
                obs=pick(t.obs),
                actions=pick(t.actions),
                rewards=pick(t.rewards),
                next_obs=pick(t.next_obs),
                terminated=pick(t.terminated),
            )
            for t in batch
        ]
        return self.learner.update(filtered)

    def save(self, path: str) -> None:
        self.learner.save(path)

    def load(self, path: str) -> None:
        self.learner.load(path)
