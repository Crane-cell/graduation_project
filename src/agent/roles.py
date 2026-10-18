# -*- coding: utf-8 -*-
"""角色分工（F11）：前锋 / 中场 / 后卫三角色定义与分配。

角色独热编码进入观测（开题 5.5）；差异化奖励偏置在 Day05/Day07 接入。
纯数据模块：零 torch / 零环境依赖。
"""
from enum import IntEnum

import numpy as np


class Role(IntEnum):
    FORWARD = 0
    MIDFIELDER = 1
    DEFENDER = 2


ROLE_LABELS = {
    Role.FORWARD: "前锋",
    Role.MIDFIELDER: "中场",
    Role.DEFENDER: "后卫",
}

# 职责一句话（任务书①：前锋进攻终结抢点、中场组织串联传球、后卫拦截协防）
ROLE_DUTIES = {
    Role.FORWARD: "进攻终结与抢点",
    Role.MIDFIELDER: "组织串联与传球",
    Role.DEFENDER: "拦截与协防",
}


def assign_roles(num_agents: int) -> list[Role]:
    """默认按 前锋→中场→后卫 循环分配（双侧队伍各自分配，前 n_per_side 为蓝方）。"""
    cycle = [Role.FORWARD, Role.MIDFIELDER, Role.DEFENDER]
    return [cycle[i % len(cycle)] for i in range(num_agents)]


def role_onehot(role: Role) -> np.ndarray:
    """角色 → 独热向量（拼入观测，F02）。"""
    vec = np.zeros(len(Role), dtype=np.float32)
    vec[int(role)] = 1.0
    return vec
