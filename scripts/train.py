# -*- coding: utf-8 -*-
"""正式训练入口（Day05）。

用法（虚拟环境）：
    F:\\A_YZY\\AProject\\venv\\Scripts\\python.exe scripts\\train.py
    F:\\A_YZY\\AProject\\venv\\Scripts\\python.exe scripts/train.py --total-steps 2000 --algorithm qmix
    F:\\A_YZY\\AProject\\venv\\Scripts\\python.exe scripts/train.py --kv algorithm.lr=0.005

覆盖项（不改配置文件）：
    --config <path>         配置文件路径（默认 src/configs/default.yaml）
    --total-steps N         覆盖 train.total_steps
    --algorithm NAME        覆盖 algorithm.name（maddpg/qmix/iql）
    --kv a.b.c=value        任意点路径覆盖（数值自动转换）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithm.heuristic import HeuristicOpponent   # noqa: E402
from src.algorithm.iql import IQL                       # noqa: E402
from src.algorithm.maddpg import MADDPG                 # noqa: E402
from src.algorithm.qmix import QMIX                     # noqa: E402
from src.algorithm.replay_buffer import ReplayBuffer    # noqa: E402
from src.env.soccer_env import SoccerEnv                # noqa: E402
from src.training.evaluator import Evaluator            # noqa: E402
from src.training.team_policy import TeamPolicy         # noqa: E402
from src.training.trainer import Trainer                # noqa: E402
from src.utils.checkpoint import CheckpointManager      # noqa: E402
from src.utils.config import load_config                # noqa: E402
from src.utils.logger import MetricLogger               # noqa: E402
from src.utils.seed import set_global_seed              # noqa: E402

ALGOS = {"maddpg": MADDPG, "qmix": QMIX, "iql": IQL}


def apply_override(cfg: dict, dotted: str) -> None:
    """--kv a.b=v：点路径覆盖（数值/布尔自动转换）。"""
    key, _, value = dotted.partition("=")
    if not value:
        raise ValueError(f"--kv 需要形如 a.b=v：{dotted!r}")
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node[p]
    try:
        node[parts[-1]] = int(value) if value.lstrip("-").isdigit() else float(value)
    except ValueError:
        node[parts[-1]] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="仿真足球 MARL 训练入口")
    parser.add_argument("--config", default=str(ROOT / "src" / "configs" / "default.yaml"))
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--algorithm", choices=list(ALGOS), default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--kv", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.total_steps is not None:
        cfg["train"]["total_steps"] = args.total_steps
    if args.algorithm is not None:
        cfg["algorithm"]["name"] = args.algorithm
    if args.eval_episodes is not None:
        cfg["eval"]["n_episodes"] = args.eval_episodes
    for kv in args.kv:
        apply_override(cfg, kv)

    algo_name = cfg["algorithm"]["name"]
    run_name = f"{algo_name}_{cfg['train']['total_steps']}steps_seed{cfg['train']['seed']}"
    set_global_seed(cfg["train"]["seed"])
    print(f"== 训练启动：{run_name} ==")

    env = SoccerEnv(cfg)
    n_blue = env.n_per_side
    learner = ALGOS[algo_name](
        num_agents=n_blue,
        obs_dim=cfg["env"]["obs_dim"],
        action_dim=cfg["env"]["action_dim"],
        config=cfg,
    )
    opponent = HeuristicOpponent(num_agents=2 * n_blue, obs_dim=cfg["env"]["obs_dim"],
                                 action_dim=cfg["env"]["action_dim"], config=cfg)
    algo = TeamPolicy(learner=learner, heuristic=opponent)

    buffer = ReplayBuffer(cfg["algorithm"]["buffer_size"])
    logger = MetricLogger(str(ROOT / cfg["logging"]["log_dir"] / run_name))
    ckpt = CheckpointManager(ROOT / cfg["logging"]["log_dir"] / run_name / "ckpt")

    trainer = Trainer(env, algo, buffer, logger, cfg, ckpt)
    result = trainer.train()
    algo.save(str(ROOT / cfg["logging"]["log_dir"] / run_name / "ckpt" / "final.pt"))

    metrics = Evaluator().evaluate(env, algo, int(cfg["eval"]["n_episodes"]), seed=cfg["eval"].get("seed"))
    print("== 训练完成 ==")
    print("train_result :", result)
    print("eval_metrics :", metrics)
    print("TensorBoard  :", f"tensorboard --logdir {ROOT / cfg['logging']['log_dir']}")
    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
