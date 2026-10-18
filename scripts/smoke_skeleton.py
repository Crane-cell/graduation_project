# -*- coding: utf-8 -*-
"""骨架冒烟脚本（Day03 分层判据现场证明——tetris 的 ConsoleMain 等价物）。

目的：在 env/algo 尚为空转骨架时，端到端跑通
  配置加载 → 环境构造 → 三算法（MADDPG/QMIX/IQL）训练主循环 → 评估协议 → 检查点
证明：①导入链无循环依赖 ②接口签名一致 ③主循环结构成立 ④架构红线不破。

用法：
    F:\\A_YZY\\AProject\\venv\\Scripts\\python.exe scripts\\smoke_skeleton.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithm.iql import IQL                       # noqa: E402
from src.algorithm.maddpg import MADDPG                 # noqa: E402
from src.algorithm.qmix import QMIX                     # noqa: E402
from src.algorithm.replay_buffer import ReplayBuffer    # noqa: E402
from src.agent.roles import assign_roles                # noqa: E402
from src.env.soccer_env import SoccerEnv                # noqa: E402
from src.training.evaluator import Evaluator            # noqa: E402
from src.training.trainer import Trainer                # noqa: E402
from src.utils.checkpoint import CheckpointManager      # noqa: E402
from src.utils.config import load_config                # noqa: E402
from src.utils.logger import MetricLogger               # noqa: E402
from src.utils.seed import set_global_seed              # noqa: E402


def check_dependency_redline() -> list[str]:
    """架构红线自检：算法层禁止 import 环境层 / 仿真框架。

    对应 tetris 教案的"Model 零 JavaFX"红线：
    算法必须与环境解耦，才能保证 MADDPG/QMIX 的通用性（QA 表 Q7 自研路线的前提）。
    """
    forbidden = (
        "from src.env", "from ..env", "from .env",
        "import pettingzoo", "from pettingzoo",
        "import gymnasium", "from gymnasium",
    )
    violations = []
    for py in sorted((ROOT / "src" / "algorithm").glob("*.py")):
        text = py.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{py.name}: {token}")
    return violations


def main() -> int:
    cfg = load_config(ROOT / "src" / "configs" / "default.yaml")
    # 冒烟覆盖：小步数快速跑通
    cfg["train"]["total_steps"] = 40
    cfg["train"]["eval_interval"] = 30
    cfg["train"]["checkpoint_interval"] = 20
    cfg["env"]["max_episode_steps"] = 10
    cfg["algorithm"]["update_start"] = 8
    set_global_seed(cfg["train"]["seed"])

    print("== 1. 配置加载 ==")
    print("algorithm.name =", cfg["algorithm"]["name"], "| seed =", cfg["train"]["seed"])

    print("== 2. 环境与角色 ==")
    env = SoccerEnv(cfg)
    print("possible_agents :", env.possible_agents)
    print("obs_space       :", env.observation_space(env.possible_agents[0]))
    print("act_space       :", env.action_space(env.possible_agents[0]))
    print("roles           :", [r.name for r in assign_roles(len(env.possible_agents))])

    buffer = ReplayBuffer(cfg["algorithm"]["buffer_size"])
    logger = MetricLogger(str(ROOT / cfg["logging"]["log_dir"] / "smoke"))
    ckpt = CheckpointManager(ROOT / cfg["logging"]["log_dir"] / "smoke" / "ckpt")

    print("== 3. 三算法 ×（训练主循环 + 评估协议）==")
    for algo_cls in (MADDPG, QMIX, IQL):
        algo = algo_cls(
            num_agents=len(env.possible_agents),
            obs_dim=cfg["env"]["obs_dim"],
            action_dim=cfg["env"]["action_dim"],
            config=cfg,
        )
        trainer = Trainer(env, algo, buffer, logger, cfg, ckpt)
        t0 = time.perf_counter()
        result = trainer.train()
        elapsed = time.perf_counter() - t0
        ev = Evaluator().evaluate(env, algo, n_episodes=2)
        print(f"{algo_cls.__name__:<7}: train={result} eval={ev} 用时 {elapsed:.2f}s")

    print("== 4. 检查点 ==")
    latest = ckpt.latest()
    print("latest checkpoint:", latest)

    print("== 5. 架构红线自检 ==")
    violations = check_dependency_redline()
    if violations:
        for v in violations:
            print("违规:", v)
        print("REDLINE: FAILED")
        logger.close()
        return 1
    print("REDLINE: OK（algorithm 层零环境依赖、零仿真框架依赖）")

    logger.close()
    print("SKELETON_SMOKE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
