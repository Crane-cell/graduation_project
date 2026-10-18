# 基于多智能体强化学习的仿真机器人足球协作系统

毕业设计课题代码仓库。在仿真足球对抗环境中，基于 CTDE（集中式训练、分散式执行）框架训练多智能体协作策略，实现 MADDPG 与 QMIX 两种主流算法，以 IQL 独立学习策略与规则基线为对照开展对抗实验。

- 学生：姚郅旸（22450150534）· 计算机与人工智能学院 · 人工智能2302班
- 课题来源：毕业设计任务书（2026-09-24 下发）
- 当前进度：Day01—Day08 完成（需求/设计/真实现/84 用例全绿·覆盖率94%/性能优化），tag 至 v0.7-perf
- GitHub：https://github.com/Crane-cell/graduation_project

## 验收指标

| 指标 | 要求 |
| --- | --- |
| 胜率 | 协作策略较 IQL 基线提升 ≥20 个百分点 |
| 团队奖励 | 收敛后平均团队奖励提升 ≥15% |
| 实时性 | 单步决策推理 ≤100 毫秒 |
| 可复现 | 训练脚本 + 超参配置 + 训练曲线（TensorBoard），固定种子可复现 |
| 文献 | 引用 ≥8 篇 |

## 目录结构

| 目录 | 职责 |
| --- | --- |
| src/env | 自研轻量 2D 足球仿真环境 SoccerEnv（PettingZoo Parallel API 兼容；Q7 实证 PettingZoo 无内置足球环境） |
| src/agent | 智能体定义与角色分工（前锋/中场/后卫） |
| src/algorithm | MADDPG、QMIX 算法实现及 IQL 基线 |
| src/training | 训练主循环、评估与课程学习 |
| src/configs | 超参数与场景配置（YAML，改配置不动代码） |
| src/utils | 日志、检查点、随机种子等工具 |
| tests | 单元测试与集成测试（pytest） |
| scripts | 训练/评估/可视化启动脚本 |
| experiments | 实验产物（训练曲线、指标表、压测数据） |
| docs | 过程文档（SRS / HLD / LLD / 环境自检 / QA 问题表 / 日报） |

## 开发环境

- Windows / Python 3.12 / PyTorch 2.6（CPU 版即可跑通全部功能；CUDA 版用于训练加速，RTX 4050 Laptop 6GB）
- 依赖清单见 `requirements.txt`（建议虚拟环境安装，不污染系统 Python）
- 环境自检报告：`docs/env_check.md`（数据来自真实运行输出）

## 快速开始

```powershell
# 1. 安装依赖（建议 Python 3.12 虚拟环境）
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
#    torch CUDA 版安装见 pytorch.org；CPU 版可运行本仓库全部功能与测试

# 2. 环境自检
python scripts/env_check.py

# 3. 训练（300 步冒烟示例；改 src/configs/default.yaml 调整规模）
python scripts/train.py --total-steps 300 --algorithm maddpg

# 4. 单元测试与覆盖率（84 用例）
python -m pytest tests --cov=src --cov-branch -q

# 5. 性能基准（NF-01 推理延迟 P50/P95、决策吞吐 loop vs batch 对比）
python scripts/benchmark.py

# 6. 可视化回放（2D 比赛动画窗口）
python scripts/visualize.py

# 训练曲线：python -m tensorboard --logdir runs
```

## Git 规范

- 分支模型：`main`（稳定基线）→ `dev`（开发集成）→ `feature/*`（各阶段工作分支）
- 提交信息：`类型: 中文描述`（feat / fix / docs / test / refactor / chore / perf）
- 里程碑 tag：`v0.0.1-skeleton` → `v0.1-requirements`（SRS）→ `v0.2-design`（HLD）→ `v0.3-lld`（LLD）→ `v0.4-core`（真实现）→ `v0.5-tests`（单测94%）→ `v0.6-data`（数据驱动）→ `v0.7-perf`（性能优化）
