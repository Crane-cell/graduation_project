# -*- coding: utf-8 -*-
"""可视化回放（Day08 附加交付）：2D 足球场俯视图，蓝队 vs 红队一局动画。

蓝队可选：heuristic（规则队，默认——可看完整对局）
        | maddpg / qmix / iql（学习算法，当前为未训练权重，对局很短；
          训练产物 M2 阶段产出后经 algo.load 挂载即可展示学习效果）。
红队 = 规则基线对手（HeuristicOpponent）。
数据采集与渲染分离：先跑完一局录轨迹，再逐帧回放（同一份轨迹可反复看）。

用法：
  python scripts/visualize.py                          # 蓝队=规则 vs 红队=规则，弹窗回放
  python scripts/visualize.py --blue maddpg --seed 7   # 蓝队=MADDPG(未训练) 看短局
  python scripts/visualize.py --interval 40 --save replay.gif
  python scripts/visualize.py --headless               # 不弹窗，仅打印轨迹统计（验证用）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

from src.algorithm.heuristic import HeuristicOpponent
from src.algorithm.iql import IQL
from src.algorithm.maddpg import MADDPG
from src.algorithm.qmix import QMIX
from src.env.soccer_env import SoccerEnv
from src.training.team_policy import TeamPolicy
from src.utils.config import load_config

ALGOS = {"maddpg": MADDPG, "qmix": QMIX, "iql": IQL}
BLUE, RED, BALL = "#1f77b4", "#d62728", "#222222"


def make_blue(blue: str, cfg: dict):
    """构造蓝队策略：heuristic=规则对手；否则为学习算法（当前未训练权重）。"""
    if blue == "heuristic":
        return HeuristicOpponent(6, 32, 10, cfg)
    return ALGOS[blue](num_agents=6, obs_dim=cfg["env"]["obs_dim"],
                       action_dim=cfg["env"]["action_dim"], config=cfg)


def collect_trajectory(cfg: dict, blue: str, seed: int, max_steps: int) -> dict:
    """跑一局并录制每步现场（位置/球/进球事件），供回放使用。"""
    env = SoccerEnv(cfg)
    policy = TeamPolicy(learner=make_blue(blue, cfg), heuristic=HeuristicOpponent(6, 32, 10, cfg))

    obs, _ = env.reset(seed=seed)
    frames: list[tuple] = []
    goals: list[tuple[int, int]] = []   # (step, goal)
    done = False
    step = 0
    while not done and step < max_steps:
        frames.append((env._pos.copy(), env._ball.copy()))
        actions = policy.select_actions_batch(obs)
        obs, _r, terms, truncs, infos = env.step(actions)
        step += 1
        g = infos[env.possible_agents[0]].get("goal", 0)
        if g:
            goals.append((step, g))
        done = all(terms.values()) or all(truncs.values()) or not env.agents

    blue_goals = sum(1 for _, g in goals if g == 1)
    red_goals = sum(1 for _, g in goals if g == 2)
    return {"frames": frames, "goals": goals, "n_steps": step,
            "blue_goals": blue_goals, "red_goals": red_goals, "seed": seed, "blue": blue}


def build_figure(cfg: dict, traj: dict):
    """构建球场静态底图 + 动态元素；返回所需 artists。"""
    w, h = cfg["env"]["field"]["width"], cfg["env"]["field"]["height"]
    cy = h / 2.0
    goal_half = cfg["env"]["goal_half_width"]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(-1, w + 1)
    ax.set_ylim(-1, h + 1)
    ax.set_aspect("equal")
    ax.set_facecolor("#eaf5e0")
    ax.grid(False)

    # 场地与球门
    ax.add_patch(Rectangle((0, 0), w, h, fill=False, edgecolor="#2c3e50", linewidth=2))
    ax.axvline(w / 2, color="#2c3e50", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(h / 2, color="#2c3e50", linestyle=":", linewidth=1, alpha=0.3)
    for x0 in (0.0, w):
        ax.add_patch(Rectangle((x0, cy - goal_half), 0.12 if x0 == 0 else -0.12,
                               goal_half * 2, fill=True, facecolor="#c9e2f7",
                               edgecolor="#2c3e50", linewidth=1.5))
    ax.text(1.5, h - 0.6, "红队（规则基线）", color=RED, fontsize=11, ha="left")
    ax.text(w - 1.5, h - 0.6, f"蓝队（{traj['blue']}）", color=BLUE, fontsize=11, ha="right")

    # 动态元素（初始占位）
    n = cfg["env"]["num_per_side"]
    blues = [Circle((0, 0), 0.3, facecolor=BLUE, edgecolor="#0b3d91", zorder=3) for _ in range(n)]
    reds = [Circle((0, 0), 0.3, facecolor=RED, edgecolor="#7a0f0f", zorder=3) for _ in range(n)]
    ball = Circle((0, 0), 0.15, facecolor=BALL, zorder=4)
    for art in blues + reds + [ball]:
        ax.add_patch(art)

    info = ax.text(0.01, -0.4, "", transform=ax.transAxes, fontsize=11,
                   bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
    return fig, blues, reds, ball, info


def animate(cfg: dict, traj: dict, interval_ms: int, save: str | None) -> None:
    frames_data = traj["frames"]
    fig, blues, reds, ball, info = build_figure(cfg, traj)

    def update(i: int):
        pos, bpos = frames_data[i]
        for k, c in enumerate(blues):
            c.set_center(pos[k])
        for k, c in enumerate(reds):
            c.set_center(pos[3 + k])
        ball.set_center(bpos)
        info.set_text(f" 步数 {i + 1}/{traj['n_steps']} | 比分 蓝 {traj['blue_goals']} : {traj['red_goals']} 红"
                      + ("  | 进球！" if any(s == i for s, _ in traj["goals"]) else ""))
        return (blues + reds + [ball, info])

    anim = FuncAnimation(fig, update, frames=len(frames_data), interval=interval_ms, blit=False, repeat=True)
    if save:
        anim.save(save, writer="pillow", fps=1000 / max(interval_ms, 1))
        print(f"[V] 动画已保存：{save}（{traj['n_steps']} 帧）")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="可视化回放：蓝队 vs 红队 2D 动画")
    ap.add_argument("--blue", choices=["heuristic", "maddpg", "qmix", "iql"], default="heuristic",
                    help="蓝队策略（默认 heuristic=规则队可看完整对局；学习算法为未训练权重、对局很短）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=500, help="回放局数上限（默认一局 500 步）")
    ap.add_argument("--interval", type=int, default=120, help="每帧毫秒数（默认 120ms/步，放慢可读；调小则更快）")
    ap.add_argument("--save", default=None, help="保存为 GIF 路径（如 out.gif）；缺省弹窗播放")
    ap.add_argument("--headless", action="store_true", help="不弹窗不保存，仅跑轨迹并打印统计（验证用）")
    args = ap.parse_args()

    cfg = load_config(ROOT / "src" / "configs" / "default.yaml")
    if args.headless:
        matplotlib.use("Agg")

    traj = collect_trajectory(cfg, args.blue, args.seed, args.max_steps)
    print(f"[V] {args.blue}（蓝）vs 规则（红）seed={args.seed}："
          f"{traj['n_steps']} 步 | 蓝 {traj['blue_goals']} : {traj['red_goals']} 红 | 进球事件 {len(traj['goals'])} 次")
    if args.headless:
        return

    animate(cfg, traj, args.interval, args.save)


if __name__ == "__main__":
    main()