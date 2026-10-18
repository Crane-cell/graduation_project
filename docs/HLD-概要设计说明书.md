# 仿真机器人足球协作系统 概要设计说明书（HLD）v1.0

- 文档版本：v1.0
- 修订日期：2026-10-18
- 编写人：姚郅旸
- 前置文档：SRS 需求规格说明书 v1.0（2026-10-18，tag v0.1-requirements）
- 评审状态：待评审

---

## 1. 概述

本设计基于 SRS v1.0，把系统拆分为 **环境层 / 智能体层 / 算法层 / 训练层 / 工具层** 五个单向依赖的包。设计目标（对应 tetris 教案"可测试、可替换、可并行"三理由）：

| 目标 | 含义 | 验证方式 |
| --- | --- | --- |
| **算法环境无关** | 算法层不 import 任何环境代码与仿真框架，保证 MADDPG/QMIX 可在任意 PettingZoo 兼容环境复用（Q7 自研环境的前提） | 架构红线自检（冒烟脚本内置） |
| **环境可替换** | 主线自研 SoccerEnv 与后续 RoboCup 2D 适配器实现同一契约 | 接口一致 + 冒烟调用链 |
| **算法可插拔** | MADDPG/QMIX/IQL 实现同一 BaseAlgo 契约，换算法只改配置 | 冒烟三算法同链路跑通 |
| **结构可测试** | 逻辑与随机源分离（种子统一注入）、组件构造注入 | Day06 pytest 全链路可测 |

**骨架冒烟已作为"分层判据现场证明"先行落地**（2.4 节，真实输出）。

## 2. 架构设计

图：`docs/diagrams/architecture.drawio`

```
scripts（入口层）
   │ 调用
   ▼
src/training 训练层（组合根：Trainer · Evaluator）
   │ 依赖（构造注入）
   ├────────────► src/env 环境层（SoccerEnv）──► src/agent（Role）──┐
   ├────────────► src/algorithm 算法层（BaseAlgo·MADDPG·QMIX·IQL·ReplayBuffer）
   └────────────► src/utils 工具层（seed·config·logger·checkpoint）◄─┘（全部层的公共底座）
```

依赖规则（单向，禁反向）：

| 规则 | 说明 |
| --- | --- |
| training → {env, agent, algorithm, utils} | 训练层是组合根，只经构造注入持有下层 |
| env → {agent, utils} + 外部 pettingzoo/gymnasium | 环境层是唯一允许接触仿真框架的层 |
| algorithm → {utils} + 外部 torch | **红线：禁止 import src.env / pettingzoo / gymnasium** |
| agent → {utils} | 纯数据模块 |
| utils → 无 | 零上层依赖的公共底座 |
| 全局禁循环依赖 | 冒烟导入链 = 无环证明 |

## 3. 包结构设计

```
src/
├── env/            # 环境层（F01）
│   └── soccer_env.py      SoccerEnv：自研足球仿真（PettingZoo Parallel API 兼容）
├── agent/          # 智能体层（F11）
│   └── roles.py           Role/assign_roles/role_onehot（纯数据）
├── algorithm/      # 算法层（F06/F07/F08/F12）——零环境依赖红线区
│   ├── base.py            BaseAlgo 契约基类
│   ├── replay_buffer.py   Transition + ReplayBuffer
│   ├── maddpg.py          MADDPG（Day05 实现网络）
│   ├── qmix.py            QMIX（Day05）
│   └── iql.py             IQL 基线（Day05）
├── training/       # 训练层（F09/F13）
│   ├── trainer.py         Trainer：collect/update 双循环主循环（结构本日定稿）
│   └── evaluator.py       Evaluator：固定局数评估协议
├── configs/        # 配置数据（F05）
│   └── default.yaml       默认配置（骨架版，Day07 数据驱动展开）
└── utils/          # 工具层
    ├── seed.py            set_global_seed（NF-02）
    ├── config.py          load_config（F05）
    ├── logger.py          MetricLogger → TensorBoard（F18）
    └── checkpoint.py      CheckpointManager（F10）
```

| 包 | 职责 | 禁止做什么 |
| --- | --- | --- |
| scripts | 入口与冒烟脚本 | 禁写业务逻辑 |
| src/training | 循环调度、评估协议 | 禁 import torch 之外的框架；禁直接操作环境内部状态 |
| src/env | 仿真物理、规则、奖励事件、观测/动作空间 | 禁 import algorithm / training |
| src/agent | 角色定义与编码 | 禁含任何训练逻辑 |
| src/algorithm | 网络结构与更新逻辑 | **禁 import env / pettingzoo / gymnasio / training** |
| src/utils | 种子/配置/日志/存档 | 禁 import 上层任何包 |

## 4. 类设计

图：`docs/diagrams/class-diagram.drawio`

| 类/接口 | 所在包 | 一句话职责 | 依赖谁 |
| --- | --- | --- | --- |
| BaseAlgo（契约） | algorithm | 算法无关接口：select_action / update / save / load | utils |
| MADDPG | algorithm | 集中式评论家+分散式演员（F07） | BaseAlgo |
| QMIX | algorithm | 个体 Q+单调混合网络（F08） | BaseAlgo |
| IQL | algorithm | 独立学习基线（F12） | BaseAlgo |
| Transition | algorithm | 全队联合转移数据类（CTDE 训练单元） | — |
| ReplayBuffer | algorithm | 转移存取与采样 | Transition |
| SoccerEnv | env | 足球仿真（PettingZoo ParallelEnv 子类） | agent.Role、pettingzoo/gymnasium |
| Role | agent | 三角色定义、分配、独热编码（F11） | — |
| Trainer | training | collect/update 主循环（F09） | env/algo/buffer/logger（注入） |
| Evaluator | training | 固定局数评估协议（F13） | env/algo（注入） |
| CheckpointManager | utils | 存档/恢复/最新查找（F10） | torch |
| MetricLogger | utils | TensorBoard 标量（F18） | torch |
| set_global_seed / load_config | utils | 种子与配置（F05/NF-02） | — |

设计要点（对应教案"当前需求撑不住的抽象一律删"）：
- 不做算法注册表工厂（3 个算法 `config["algorithm"]["name"]` 直接映射即可，新增算法时再抽）；
- Role 是 IntEnum 数据而非类层次（差异是数据不是行为）；
- Trainer 只依赖 BaseAlgo 契约（面向接口，同 tetris 的 GameController 只认 GameModel 接口）。

## 5. 接口设计

### 5.1 BaseAlgo 契约（F06，Trainer 视角）

| 方法 | 参数 | 返回 | 语义 | 对应需求 |
| --- | --- | --- | --- | --- |
| select_action | (agent_id: str, obs: ndarray) | int | 单智能体观测 → 离散动作（训练期含探索） | F02/F03 |
| update | (batch: list[Transition]) | dict[str, float] | 一批联合转移 → 网络更新 → 损失字典 | F07/F08/F12 |
| save / load | (path: str) | None | 算法状态落盘/恢复 | F10 |

### 5.2 SoccerEnv 契约（F01，PettingZoo ParallelEnv 兼容）

| 方法/属性 | 返回 | 语义 | 对应需求 |
| --- | --- | --- | --- |
| reset(seed) | (obs_dict, infos_dict) | 重置一局 | F01 |
| step(actions_dict) | (obs, rewards, terms, truncs, infos) | 全队同步推进一帧 | F01/F04 |
| observation_space(a) / action_space(a) | Box / Discrete | 空间声明 | F02/F03 |
| possible_agents | list[str] | blue_N… / red_N… | F01/F11 |

### 5.3 ReplayBuffer 契约

| 方法 | 语义 |
| --- | --- |
| add(Transition) | 存入全队联合转移 |
| sample(batch_size, rng) | 随机采样（传入固定 rng 保可复现） |
| __len__ | 当前容量（Trainer 用它判断 update_start 阈值） |

## 6. 动态设计

### 6.1 时序图 1：一次采集步（collect）

图：`docs/diagrams/sequence-collect.drawio`。消息对照代码：Trainer.train() 中 `{a: algo.select_action(a, obs[a])}` → `env.step(actions)` → `buffer.add(Transition(...))` → 局末 `env.reset()`（与 SRS 活动图 7.1 的 episode 内循环一致）。

### 6.2 时序图 2：一次网络更新（update）

图：`docs/diagrams/sequence-update.drawio`。消息对照代码：Trainer.train() 更新分支（update_start / update_every 双阈值控制）→ `buffer.sample(batch)` → `algo.update(batch)` → `logger.write(...)`。算法内部（2a 自消息）Day05 由各算法实现。

### 6.3 数据流图

图：`docs/diagrams/dataflow.drawio`。观察角度：时序图回答"谁调用谁"；数据流图回答"配置、动作、转移、参数、曲线"这些**数据**怎么流（含 D2 参数回边 P1 的闭环）。

## 7. 数据设计

| 数据 | 类型/结构 | 归属 | 谁可写 |
| --- | --- | --- | --- |
| 联合转移 | Transition{obs, actions, rewards, next_obs, terminated}（均为 dict） | ReplayBuffer | 仅 Trainer 采集分支 |
| 配置 | 嵌套 dict（YAML：env/algorithm/train/eval/logging 五节） | configs/*.yaml | 研究者手改，程序只读 |
| 网络参数 | torch state_dict（含目标网络） | 各算法实例 | 仅算法 update 分支 |
| 检查点 | {saved_at, global_step, **state}（torch.save） | runs/**/ckpt | CheckpointManager |
| 训练标量 | TensorBoard 事件文件 | runs/ | 仅 MetricLogger |

## 8. 需求-设计跟踪矩阵

| 需求 | 对应设计元素 | 体现处 | 状态 |
| --- | --- | --- | --- |
| F01 环境封装 | env.SoccerEnv（ParallelEnv 契约） | 架构图/类图/时序图1 | 骨架已落地 |
| F02 观测编码 | SoccerEnv.observation（编码器 Day05 拆出） | 类图 | 结构预留 |
| F03 动作接口 | SoccerEnv.action_space/step 映射 | 类图/时序图1 | 骨架已落地 |
| F04 奖励计算 | SoccerEnv.step 奖励事件（塑形器 Day05） | 数据流图 P2 | 结构预留 |
| F05 配置外置 | utils.config + configs/default.yaml | 数据流图 D3 | 骨架已落地 |
| F06 智能体网络 | BaseAlgo 契约（网络 Day05） | 类图 | 契约已落地 |
| F07 MADDPG | algorithm.maddpg.MADDPG | 类图 | 骨架占位 |
| F08 QMIX | algorithm.qmix.QMIX | 类图 | 骨架占位 |
| F09 训练主循环 | training.trainer.Trainer | 时序图1+2、数据流图 | **结构定稿** |
| F10 检查点存取 | utils.checkpoint.CheckpointManager | 架构图/数据流图 D4 | 最小版已落地 |
| F11 角色分工 | agent.roles（Role/assign_roles/role_onehot） | 类图 | 已落地 |
| F12 IQL 基线 | algorithm.iql.IQL | 类图 | 骨架占位 |
| F13 对抗评估协议 | training.evaluator.Evaluator + eval 配置节 | 类图、SRS 活动图 7.2 | 结构定稿（统计 Day05 实数化） |
| F14 协作塑形 | 并入 env 奖励事件（塑形项 Day05 定义） | 数据流图 P2 | [可延迟 Day05] |
| F15 信用分配 | algorithm 层扩展点（反事实/差分奖励，Day04 详化） | — | [可延迟 Day04] |
| F16 有限通信 | algorithm 层扩展点（Day04 详化） | — | [可延迟 Day04] |
| F17 消融支持 | YAML 开关字段（Day07 展开） | — | [可延迟 Day07] |
| F18 曲线可视化 | utils.logger.MetricLogger | 架构图/数据流图 P5 | 已落地 |
| F19 课程学习 | 训练层扩展点（场景配置化已预留） | — | [可延迟] |
| F20 对手池 | 训练层扩展点 | — | [可延迟] |
| F21 RoboCup 2D | env 层适配器（ParallelEnv 契约保证可替换） | 架构图 | [可延迟] |
| F22 明确不做 | 无对应设计（正确——不做的不设计） | — | — |

自检：F01—F13 每条有设计接住（无遗漏）；无 F14 之外的多余功能（无过度设计）。✓

## 9. 附录 A：AI 使用与核对说明 + 分层判据现场证明

设计推导链（对应教案示范⑦）：**SRS 活动图 → 时序图 → 类图**。本 HLD 的类图方法均可在时序图消息中找到对应；时序图消息均对应骨架代码的真实调用（无"图上有的方法代码里没有"）。

**骨架冒烟（现场证明）**——`scripts/smoke_skeleton.py` 于 2026-10-18 真实运行（退出码 0，节选）：

```
possible_agents : ['blue_0', 'blue_1', 'blue_2', 'red_0', 'red_1', 'red_2']
obs_space       : Box(-1.0, 1.0, (32,), float32)
act_space       : Discrete(10)
MADDPG : train={'global_step': 40, 'buffer_size': 40} eval={...} 用时 0.01s
QMIX   : train={'global_step': 40, 'buffer_size': 80} eval={...} 用时 0.01s
IQL    : train={'global_step': 40, 'buffer_size': 120} eval={...} 用时 0.01s
latest checkpoint: latest
REDLINE: OK（algorithm 层零环境依赖、零仿真框架依赖）
SKELETON_SMOKE_OK
```

冒烟证明三件事：①导入链无循环依赖；②接口签名一致（三算法同链路可跑）；③主循环结构成立（40 步 collect→add→update→reset→存档全链路）。骨架期评估指标为零值占位——正式指标须待 Day05 真实实现后由评估协议产出（不允许把占位值写入任何报告）。

AI 核对五项（架构图/类图/时序图/数据流图均按本项自查）：无过度设计 ✓ / 无循环依赖 ✓ / 图码一致 ✓ / 术语统一（SRS 第 2 章）✓ / 红线经脚本实证 ✓。

## 10. 附录 B：AI 假设清单（逐条处置）

| 序号 | AI 假设 | 处置 | 依据 |
| --- | --- | --- | --- |
| 1 | MADDPG 离散动作适配用 Gumbel-Softmax | 待确认（登记 Q8，Day04 定稿） | MADDPG 原文离散化实验 |
| 2 | QMIX 混合网络的全局状态用全体观测拼接 | 待确认（登记 Q9） | MARL 常规做法 |
| 3 | 3v3 场地 20×15m、episode ≤500 步等物理参数 | 待确认（登记 Q10，YAML 已标 [占位]） | 骨架默认可配置 |
| 4 | 评估协议 inference_ms_p50 以整队一次批量前向计时 | 采纳（SRS NF-01 已定口径，Q4 汇总口径） | 教案"测量驱动" |
| 5 | 三个算法共用 Trainer/环境契约即可公平对比，无需算法注册表 | 采纳（删过度设计） | SRS F06/F12/F13 |

---

*文档结束。评审通过后版本升至 v1.1 并同步 Git tag v0.2-design。*
