# 环境自检报告（env_check）

- 自检日期：2026-10-18
- 自检脚本：`scripts/env_check.py`（本报告数据的**唯一来源**，可随时复跑）
- 复跑命令：`F:\A_YZY\AProject\venv\Scripts\python.exe scripts\env_check.py`
- 结论：**ENV_CHECK: PASSED**（退出码 0）

---

## 1. 运行时与依赖版本（真实输出）

| 项目 | 版本 / 值 | 说明 |
| --- | --- | --- |
| Python | 3.12.14 | 满足课题 3.10+ 要求 |
| 解释器路径 | `F:\A_YZY\AProject\venv\Scripts\python.exe` | 虚拟环境，不污染系统 Python |
| torch | 2.6.0+cpu | CUDA 版 2.6.0 安装包同步下载中（见第 4 节） |
| cuda 可用 | False（当前为 CPU 版） | 切换 CUDA 后需复跑本脚本核对 |
| numpy | 1.26.4 | 锁定 <2 以兼容 pettingzoo |
| gymnasium | 1.4.0 | — |
| pettingzoo | 1.27.0 | 满足 requirements 的 ≥1.24 |
| pyyaml | 6.0.3 | 超参配置外置用（Day07） |
| matplotlib | 3.11.2 | 实验图表用 |
| pytest | 9.1.1 | 单元测试用（Day06） |
| pytest-cov | 7.1.0 | 覆盖率统计用（Day06） |
| tensorboard | 2.21.0 | 训练曲线用（F18） |
| Git | 2.55.0.windows.3（本地功能正常） | 网络 remote 助手缺失，推送经 GitHub Desktop |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6141 MiB | CUDA 版 torch 的目标设备 |

## 2. 多智能体仿真框架连通性（真实输出）

以 pettingzoo 1.27.0 内置的 `cooperative_pong_v6` 做 AEC 冒烟测试：

```
cooperative_pong_v6 : OK（reset + 推进 10 步）
num_agents        : 2
possible_agents   : ['paddle_0', 'paddle_1']
obs_space         : Box(0, 255, (280, 480, 3), uint8)
act_space         : Discrete(3)
```

多智能体 AEC 框架（reset / agent_iter / step / action_space / observation_space）在本机工作正常——自研足球环境（F01）将实现同一套 API。

## 3. 张量前向冒烟测试（真实输出）

```
forward      : OK, 输出形状 (32, 8)
```

## 4. 两条实证发现（重要）

1. **pettingzoo 1.27.0 无内置足球环境**：遍历其全部环境家族目录（atari / butterfly / classic / magent / mpe / sisl），检索 `soccer` / `football` 关键词**零命中**。开题报告 5.2 节"PettingZoo（含足球环境封装）"表述与实际不符。
   - **处置**：足球仿真环境改为**自研**（PettingZoo AEC/Parallel API 兼容），恰好对应任务书①"仿真环境建模与协作策略总体设计"的核心工作；已登记 QA 表 **Q7**，待导师确认；SRS F01 已同步修订。
2. **MPE 家族已从 pettingzoo 拆分为独立包 mpe2**（`pip install mpe2`）。
   - **处置**：Day05 算法验证阶段按需安装 mpe2，可作为 MADDPG 实现正确性的标准对照环境（MADDPG 原论文实验环境）。

**CUDA 版 torch 说明**：当前 venv 内为 CPU 版 2.6.0（Day01—Day08 全部工作可运行，所有数字均真实有效）；CUDA 版 2.6.0+cu124 安装包（2532 MB）正以后台断点续传方式下载至 `F:\A_YZY\AProject\wheels\`，完成后执行本地切换（同版本号，零代码风险），服务毕设 GPU 训练。

## 5. 已知问题与规避

| 问题 | 规避方式 |
| --- | --- |
| Git 网络 remote-https 助手缺失（F:\A_YZY\Git\Git 安装不完整） | 本地 init/commit/tag 均正常；推送 GitHub 使用 GitHub Desktop |
| pip 直连 download.pytorch.org 曾发生下载停滞（约 140MB 处连接冻结） | CUDA 包改用 curl 断点续传 + 卡死自动重连循环下载 |
| cooperative_pong_v0 已弃用 | 使用 cooperative_pong_v6 |

---

*本报告所有数据来自 2026-10-18 实际运行输出，未使用任何 AI 估算值。*
