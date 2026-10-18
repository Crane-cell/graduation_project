# -*- coding: utf-8 -*-
"""跨层共享常量（utils 是公共底座，env 与 algorithm 均可 import）。

动作空间（LLD 第 2 章）与观测布局（LLD 第 3 章）的唯一权威定义；
env 与 heuristic/测试均引用此处，杜绝两处表格漂移。
"""
import numpy as np

# ---------- 动作空间（action_dim = 10） ----------
ACTION_DIM = 10
ACTION_KICK = 8   # 控球时沿朝向踢球
ACTION_STAY = 9   # 原地不动
MOVE_ACTION_COUNT = 8  # 动作 0—7 为八方向移动

# 八方向单位向量（顺序 = 动作编号 0—7）：E/NE/N/NW/W/SW/S/SE
DIRECTIONS_8 = np.array(
    [
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 0.0],
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
    ],
    dtype=np.float32,
)
# 归一化（NE 等对角线方向长度为 1）
DIRECTIONS_8 /= np.linalg.norm(DIRECTIONS_8, axis=1, keepdims=True)

# ---------- 观测布局偏移（obs_dim = 32，LLD 第 3 章） ----------
OBS_SELF_POS = 0        # 0-1   自身位置（归一化）
OBS_HEADING = 2          # 2-3   朝向单位向量
OBS_BALL_POS = 4         # 4-5   球位置（全局）
OBS_BALL_REL = 6         # 6-7   球相对自身
OBS_BALL_VEL = 8         # 8-9   球速度
OBS_TEAMMATE_REL = 10    # 10-15 队友相对位置（2 人 × 2）
OBS_OPPONENT_REL = 16    # 16-21 对手相对位置（3 人 × 2）
OBS_ROLE_ONEHOT = 22     # 22-24 角色独热
OBS_TEAM_ID = 25         # 25    队伍标识（蓝 +1 / 红 -1）
OBS_CONTROL_SELF = 26    # 26    自身是否控球
OBS_CONTROL_TEAM = 27    # 27    己方是否控球
OBS_ATTACK_GOAL_REL = 28  # 28-29 进攻门中心相对位置
OBS_DEFEND_GOAL_REL = 30   # 30-31 防守门中心相对位置
OBS_DIM = 32
