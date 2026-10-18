# -*- coding: utf-8 -*-
"""Day08 性能压测脚本：训练吞吐 / 决策吞吐 / NF-01 推理延迟 / cProfile 热点。

场景：
  A 训练满负荷吞吐  ：真实 Trainer 主循环（采集+更新），报告 ms/步、步/秒
  B 决策吞吐        ：--mode loop（逐智能体单样本前向）vs --mode batch（整队批量前向）
                      —— 优化前后对比的隔离口径
  C NF-01 推理延迟   ：Evaluator 同口径（整队一次决策耗时 P50/P95），对照 100ms 指标
  --profile         ：对场景 B 做 cProfile，输出累积耗时 Top15（热点定位）

用法示例：
  python scripts/benchmark.py                     # 默认 maddpg，三场景全跑
  python scripts/benchmark.py --algo qmix --steps 200
  python scripts/benchmark.py --mode batch --decisions 2000
  python scripts/benchmark.py --profile
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithm.heuristic import HeuristicOpponent
from src.algorithm.iql import IQL
from src.algorithm.maddpg import MADDPG
from src.algorithm.qmix import QMIX
from src.algorithm.replay_buffer import ReplayBuffer
from src.env.soccer_env import SoccerEnv
from src.training.evaluator import Evaluator
from src.training.team_policy import TeamPolicy
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import MetricLogger

ALGOS = {"maddpg": MADDPG, "qmix": QMIX, "iql": IQL}


def build(cfg: dict, name: str):
    env = SoccerEnv(cfg)
    algo = ALGOS[name](num_agents=len(env.possible_agents), obs_dim=cfg["env"]["obs_dim"],
                       action_dim=cfg["env"]["action_dim"], config=cfg)
    return env, algo


def scenario_a_train_throughput(cfg: dict, name: str, steps: int) -> None:
    """真实训练主循环吞吐（采集+决策+更新），产出 ms/step 与步/秒。"""
    env, algo = build(cfg, name)
    cfg_t = dict(cfg)
    cfg_t["train"] = {"total_steps": steps, "eval_interval": 10**9, "checkpoint_interval": 10**9}
    buffer = ReplayBuffer(int(cfg_t["algorithm"]["buffer_size"]))
    logger = MetricLogger(ROOT / "runs" / f"bench-{name}")
    trainer = Trainer(env, algo, buffer, logger, cfg_t, checkpoint_manager=None)
    t0 = time.perf_counter()
    result = trainer.train()
    wall = time.perf_counter() - t0
    logger.close()
    print(f"[A] 训练吞吐（{name}, {result['global_step']} 步）")
    print(f"    总耗时 {wall:.2f}s | 平均 {wall / steps * 1000:.2f} ms/步 | {steps / wall:.0f} 步/秒")


def scenario_b_decision_throughput(cfg: dict, name: str, decisions: int, mode: str) -> None:
    """决策吞吐：loop（逐智能体×6 次前向）vs batch（整队一次前向）。"""
    env, algo = build(cfg, name)
    obs = {a: np.zeros(cfg["env"]["obs_dim"], dtype=np.float32) for a in env.possible_agents}
    if mode == "loop":
        def one_step():
            return {a: algo.select_action(a, obs[a]) for a in obs}
    else:
        def one_step():
            return algo.select_actions_batch(obs)

    for _ in range(50):  # 预热（懒初始化/缓存）
        one_step()
    t0 = time.perf_counter()
    for _ in range(decisions):
        one_step()
    wall = time.perf_counter() - t0
    per = wall / decisions * 1000
    print(f"[B] 决策吞吐量（{name} / {mode}）{decisions} 次整队决策")
    print(f"    平均 {per:.3f} ms/次整队决策 | {decisions / wall:.0f} 次/秒")


def scenario_c_nf01(cfg: dict, name: str, n_episodes: int) -> None:
    """NF-01 同口径：蓝队学习 + 红队规则，整队决策延迟 P50/P95（对照 100ms）。"""
    env, algo = build(cfg, name)
    policy = TeamPolicy(learner=algo, heuristic=HeuristicOpponent(6, 32, 10, cfg))
    metrics = Evaluator().evaluate(env, policy, n_episodes=n_episodes, seed=2026)
    p50, p95 = metrics["inference_ms_p50"], metrics["inference_ms_p95"]
    verdict = "PASS（≤100ms）" if p95 <= 100.0 else "FAIL（>100ms）"
    print(f"[C] NF-01 推理延迟（{name}，{int(metrics['n_episodes'])} 局）")
    print(f"    整队决策 P50 = {p50:.3f} ms | P95 = {p95:.3f} ms → {verdict}")
    print(f"    同局统计：胜率 {metrics['win_rate']:.2f} / 负率 {metrics['loss_rate']:.2f} / 场均净胜球 {metrics['goal_diff_per_game']:.2f}")


def scenario_profile(cfg: dict, name: str, decisions: int) -> None:
    """cProfile 热点：对场景 B 决策循环做剖析，输出累积耗时 Top15。"""
    env, algo = build(cfg, name)
    obs = {a: np.zeros(cfg["env"]["obs_dim"], dtype=np.float32) for a in env.possible_agents}

    def workload():
        for _ in range(decisions):
            algo.select_actions_batch(obs)

    prof = cProfile.Profile()
    prof.enable()
    workload()
    prof.disable()
    stats = pstats.Stats(prof)
    stats.sort_stats("cumulative")
    print(f"[P] cProfile 热点 Top15（{name} / batch 模式 / {decisions} 次整队决策）")
    stats.print_stats(15)


def main() -> None:
    ap = argparse.ArgumentParser(description="Day08 性能压测")
    ap.add_argument("--algo", choices=sorted(ALGOS), default="maddpg")
    ap.add_argument("--steps", type=int, default=400, help="场景A 训练步数")
    ap.add_argument("--decisions", type=int, default=2000, help="场景B 决策次数")
    ap.add_argument("--mode", choices=["loop", "batch"], default="loop", help="场景B 决策路径")
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--profile", action="store_true", help="附带 cProfile 热点剖析")
    args = ap.parse_args()

    print("== 环境 ==")
    print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | "
          f"threads={torch.get_num_threads()}")

    cfg = load_config(ROOT / "src" / "configs" / "default.yaml")
    print(f"算法={args.algo} | 3v3 | obs_dim={cfg['env']['obs_dim']} | hidden={cfg['algorithm']['hidden_dim']}")

    if args.profile:
        scenario_profile(cfg, args.algo, args.decisions)
        return

    scenario_a_train_throughput(cfg, args.algo, args.steps)
    scenario_b_decision_throughput(cfg, args.algo, args.decisions, args.mode)
    scenario_c_nf01(cfg, args.algo, args.eval_episodes)


if __name__ == "__main__":
    main()