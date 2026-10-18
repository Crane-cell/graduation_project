# -*- coding: utf-8 -*-
"""环境自检脚本：验证开发环境完整性（Day01 交付，数据真实输出的唯一来源）。

用法（虚拟环境）：
    F:\\A_YZY\\AProject\\venv\\Scripts\\python.exe scripts\\env_check.py
"""
import sys


def main() -> int:
    failures = []

    print("== 1. Python 运行时 ==")
    print("python :", sys.version.split()[0])
    print("exe    :", sys.executable)
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        failures.append(f"Python {major}.{minor} < 3.10")

    print("== 2. 核心依赖版本 ==")
    import torch
    print("torch        :", torch.__version__)
    print("cuda可用     :", torch.cuda.is_available())

    import numpy
    print("numpy        :", numpy.__version__)

    import gymnasium
    print("gymnasium    :", gymnasium.__version__)

    import pettingzoo
    print("pettingzoo   :", pettingzoo.__version__)

    import yaml
    print("pyyaml       :", yaml.__version__)

    import matplotlib
    print("matplotlib   :", matplotlib.__version__)

    import pytest
    print("pytest       :", pytest.__version__)

    print("== 3. 多智能体仿真框架连通性（AEC 冒烟测试） ==")
    # 注：pettingzoo 1.27.0 无内置足球环境，且 MPE 已拆分为独立包 mpe2（均已于 2026-10-18 实证，
    #     见 docs/QA问题表.md Q7）。足球仿真环境为本课题自研交付（F01，PettingZoo API 兼容），
    #     此处以内置的 cooperative_pong 验证"多智能体 AEC 框架可正常 reset/推进"。
    try:
        from pettingzoo.butterfly import cooperative_pong_v6

        env = cooperative_pong_v6.env()
        env.reset(seed=42)
        steps = 0
        for agent in env.agent_iter(max_iter=10):
            obs, reward, term, trunc, info = env.last()
            action = None if (term or trunc) else env.action_space(agent).sample()
            env.step(action)
            steps += 1
        print("cooperative_pong_v6 : OK（reset + 推进 %d 步）" % steps)
        print("num_agents        :", env.num_agents)
        print("possible_agents   :", env.possible_agents)
        print("obs_space         :", env.observation_space(env.possible_agents[0]))
        print("act_space         :", env.action_space(env.possible_agents[0]))
    except Exception as exc:  # noqa: BLE001
        failures.append(f"pettingzoo 多智能体框架异常: {exc!r}")
        print("cooperative_pong_v6 : FAIL ->", repr(exc))

    print("== 4. 张量前向冒烟测试 ==")
    try:
        net = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8))
        x = torch.randn(32, 32)
        y = net(x)
        print("forward      : OK, 输出形状", tuple(y.shape))
        if torch.isnan(y).any():
            failures.append("前向输出含 NaN")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"张量前向异常: {exc!r}")

    print("== 5. 结论 ==")
    if failures:
        for f in failures:
            print("FAIL:", f)
        print("ENV_CHECK: FAILED")
        return 1
    print("ENV_CHECK: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
