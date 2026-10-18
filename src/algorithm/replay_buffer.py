# -*- coding: utf-8 -*-
"""经验回放（F09 依赖组件）：全队联合转移的存储与采样。

数据结构说明（HLD 数据设计）：MARL 的转移是"全队同步"的四元组——
obs/actions/rewards/next_obs 均为 {agent_id: 值} 字典，
CTDE 算法（MADDPG 集中评论家 / QMIX 混合网络）需要整队信息做更新。
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


@dataclass
class Transition:
    """全队联合转移（CTDE 训练单元）。"""

    obs: dict          # {agent_id: np.ndarray}
    actions: dict      # {agent_id: int}
    rewards: dict      # {agent_id: float}
    next_obs: dict     # {agent_id: np.ndarray}
    terminated: dict   # {agent_id: bool}


class ReplayBuffer:
    """环形缓冲：容量满后自动淘汰最旧转移。"""

    def __init__(self, capacity: int) -> None:
        self._buf: deque[Transition] = deque(maxlen=int(capacity))

    def add(self, transition: Transition) -> None:
        self._buf.append(transition)

    def sample(self, batch_size: int, rng: random.Random | None = None) -> list[Transition]:
        """随机采样一批转移；传入固定种子的 rng 保证可复现。"""
        if len(self._buf) < batch_size:
            return list(self._buf)
        rng = rng or random
        indices = rng.sample(range(len(self._buf)), batch_size)
        return [self._buf[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buf)
