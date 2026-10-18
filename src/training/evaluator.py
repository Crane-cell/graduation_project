# -*- coding: utf-8 -*-
"""对抗评估协议（F13，Day05 实数化）：固定局数统计真实指标。

指标口径（蓝队=学习队视角）：
  win_rate / goal_diff（场均净胜球）/ team_reward（蓝队局回报均值）
  / inference_ms_p50 与 p95（整队一次决策耗时，NF-01 同口径）
数据全部来自本函数真实运行输出——报告引用须注明局数与种子。
"""
from __future__ import annotations

import time

import numpy as np


class Evaluator:
    def evaluate(self, env, algo, n_episodes: int, seed: int | None = None) -> dict[str, float]:
        wins = draws = losses = 0
        goal_diff = 0
        ep_returns: list[float] = []
        inference_ms: list[float] = []

        for ep in range(n_episodes):
            ep_seed = None if seed is None else seed + ep
            obs, _ = env.reset(seed=ep_seed)
            done = False
            ep_reward_blue = 0.0
            last_goal = 0
            while not done:
                t0 = time.perf_counter()
                actions = algo.select_actions_batch(obs)
                inference_ms.append((time.perf_counter() - t0) * 1000.0)
                obs, rewards, terms, truncs, infos = env.step(actions)
                blue_ids = [a for a in rewards if a.startswith("blue")]
                ep_reward_blue += sum(rewards[a] for a in blue_ids) if blue_ids else 0.0
                last_goal = infos.get(env.possible_agents[0], {}).get("goal", last_goal)
                done = all(terms.values()) or all(truncs.values()) or not env.agents

            ep_returns.append(ep_reward_blue)
            if last_goal == 1:
                wins += 1
                goal_diff += 1
            elif last_goal == 2:
                losses += 1
                goal_diff -= 1
            else:
                draws += 1

        n = max(n_episodes, 1)
        return {
            "win_rate": wins / n,
            "loss_rate": losses / n,
            "draw_rate": draws / n,
            "goal_diff_per_game": goal_diff / n,
            "team_reward": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "inference_ms_p50": float(np.percentile(inference_ms, 50)) if inference_ms else 0.0,
            "inference_ms_p95": float(np.percentile(inference_ms, 95)) if inference_ms else 0.0,
            "n_episodes": float(n_episodes),
        }
