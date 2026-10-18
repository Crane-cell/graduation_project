# 仿真机器人足球协作系统 详细设计说明书（LLD）v1.0

- 文档版本：v1.0
- 修订日期：2026-10-18
- 编写人：姚郅旸
- 前置文档：HLD 概要设计说明书 v1.0（tag v0.2-design）
- 评审状态：待评审

---

## 1. 概述

本 LLD 把 HLD 的五层骨架细化到**方法级**：动作空间与观测编码定稿（解开 SRS F02/F03 与 QA Q10 占位）、环境物理与规则定稿、三种算法的网络结构与更新公式定稿（解开 Q8/Q9）、训练状态机、方法级设计表。所有表中的默认参数即 `src/configs/default.yaml` 的真实取值；YAML 中的 [占位] 标注随本设计解除。

## 2. 动作空间定稿（F03，action_dim=10）

| 动作编号 | 语义 |
| --- | --- |
| 0—7 | 八方向移动（E/NE/N/NW/W/SW/S/SE），每步位移 1.0 m，并把朝向更新为该方向单位向量 |
| 8 | 踢球：**控球**（与球距离 < 0.5 m）时沿当前朝向把球踢出，球速 3.0 m/步；未控球时该动作无效果 |
| 9 | 原地不动（朝向保持） |

设计理由：踢球沿朝向而非自由选向——传球需要"先跑位对准、再踢"，协作行为（接应跑位）由此涌现，这正是任务书②要研究的协作要素；动作数固定 10 与冒烟契约一致。

## 3. 观测编码定稿（F02，obs_dim=32，3v3）

| 偏移 | 内容 | 归一化 |
| --- | --- | --- |
| 0—1 | 自身位置 (x, y) | ÷场地半宽/半高 → [-1,1] |
| 2—3 | 自身朝向单位向量 | 单位圆 |
| 4—5 | 球位置（全局可见——球是全场焦点，符合足球事实） | 同上 |
| 6—7 | 球相对自身位置 | ÷场地尺度 |
| 8—9 | 球速度 | ÷3.0 上限 |
| 10—15 | 队友 1/2 相对位置 | 同上 |
| 16—21 | 对手 1/2/3 相对位置 | 同上 |
| 22—24 | 角色独热（前锋/中场/后卫，F11） | 0/1 |
| 25 | 队伍标识 | 蓝=1.0 / 红=-1.0 |
| 26 | 是否控球（自身） | 0/1 |
| 27 | 己方是否控球（含队友） | 0/1 |
| 28—29 | 进攻门中心相对位置 | 同上 |
| 30—31 | 防守门中心相对位置 | 同上 |

合计 32 维，全部取值落在 [-1,1]（SRS GWT-2 可测）。**部分可观测保留**：无对方速度、无全局上帝视角，Q9 的拼接观测仅用于训练期混合网络（CTDE），执行期仍是本观测。

## 4. 环境物理与规则定稿（Q10 解除）

| 项 | 取值（YAML 可改） |
| --- | --- |
| 场地 | 宽 20.0 m × 高 15.0 m |
| 球门 | 左右两侧，门高 4.0 m（y ∈ [5.5, 9.5]） |
| 智能体/球半径 | 0.3 m / 0.15 m |
| 移动速度 | 1.0 m/步 |
| 踢球球速 | 3.0 m/步，球摩擦 0.9/步 |
| 智能体碰撞 | 重叠时沿连线各推回半程 |
| 球-智能体 | 距离 < 0.45 m 时球被推出并继承该智能体 30% 移动速度 |
| 控球 | 与球距离 < 0.5 m |
| 边界 | 上下墙反弹（速度分量取反）；球出左右边界仅在门区判进球，否则贴墙 |
| 进球 | 球心 x<0 且 y∈门区 → 红方得分；x>20 且 y∈门区 → 蓝方得分；**进球立即终局** |
| 终局 | 进球或 500 步 → 全体 terminated=True |
| 初始站位 | 蓝方 x∈[2,6]、红方 x∈[14,18] 随机 y；球置中心 |
| 种子 | reset(seed) → np.random.default_rng(seed)，全物理随机仅经此 rng |

奖励定稿（F04；蓝队视角，红队对称；权重进 YAML `env.reward`）：

| 事件 | 值 |
| --- | --- |
| 进球（团队稀疏） | +10.0 |
| 失球 | -10.0 |
| 控球（每步塑形） | +0.02 |
| 球向对方门位移 | +0.5 × Δx进攻方向 |
| 球向己方门位移 | -0.5 × Δx |
| 与球距离缩小（跑向球） | +0.005 × 缩小量 |

塑形项之和远小于稀疏奖励，防奖励投机（SRS 5.3 防投机要求；消融开关 Day07）。

## 5. 算法与对手定稿（Q8/Q9 解除）

**队伍编排（对齐任务书③）**：训练与评估固定为 **蓝队=学习智能体（3 人），红队=规则基线对手**。`TeamPolicy`（训练层组合器，实现 BaseAlgo 契约）：`select_action` 按队伍前缀分发——蓝队走 learner、红队走 HeuristicOpponent；`update` 只把蓝队的 obs/actions/rewards/terminated 过滤转交 learner；`save/load` 委托 learner。Trainer/契约零改动。

**HeuristicOpponent（规则基线）**：无内部状态、纯观测函数——己方距球最近者朝球移动；控球者若朝向已对准进攻门则踢（动作 8），否则选朝门方向的移动动作对准；其余球员朝球移动。对应任务书③"规则基线队伍"。

**MADDPG（Q8 解除：Gumbel-Softmax 离散化）**，n=蓝队 3 人：

| 组件 | 结构 |
| --- | --- |
| actor_i | MLP(32→64→64→10)，输出 logits |
| critic_i | MLP(3×32 + 3×10 =126 →64→64→1) |
| 执行 | ε-greedy（ε=0.1）：以 ε 随机动作，否则 argmax(logits) |
| 目标网络 | actor'/critic' 软更新（tau_soft=0.01） |

训练（一批 B=32 个联合转移；ã/â = Gumbel-Softmax(ST, τ=1.0) 硬 one-hot）：
- critic：y_i = r_i + γ(1-d)·Q'_i(o', â')，Loss_c = MSE(Q_i(o, a), y_i)
- actor：Loss_a = -E[Q_i(o, ã_i)]，ã_i 仅对第 i 维可导（ST 直通）
- 返回 {"critic_loss", "actor_loss"}

**QMIX（Q9 解除：拼接观测作全局状态）**：

| 组件 | 结构 |
| --- | --- |
| q_i | MLP(32→64→64→10) ×3 |
| 状态 s | 全体蓝队观测拼接 96 维（仅训练期使用，CTDE） |
| Mixer | w₁=abs(hyper_w1(s))∈R^{3×32}，b₁=hyper_b1(s)；h=ELU(Σw₁q+b₁)；w₂=abs(hyper_w2(s))，b₂=hyper_b2(s)；Q_tot=Σw₂h+b₂ |
| 目标网络 | q'/mixer' 软更新 |

- r_team = Σ 蓝队奖励；y = r_team + γ(1-d)·Q_tot'(s', argmax q')（单调性保证个体 argmax 即联合最优）
- Loss = MSE(Q_tot, y)；返回 {"td_loss"}

**IQL 基线（F12）**：q_i = MLP(32→64→64→10) 独立 DQN，y_i = r_i + γ(1-d)·max q'_i(o')，ε-greedy。

## 6. 训练状态机（F09）

图：`docs/diagrams/statemachine-training.drawio`（主状态机含 TRAINING 复合子状态机）

| 源 | 触发[守卫] | 目标 | 动作 |
| --- | --- | --- | --- |
| ● | train() 调用 | INIT | 加载配置、set_global_seed |
| INIT | 组件构造成功 | TRAINING | env.reset |
| TRAINING entry | — | COLLECT | — |
| COLLECT | len(buf) ≥ update_start ∧ step%update_every==0 | UPDATE | 采样→algo.update→写损失 |
| COLLECT / UPDATE | episode 结束 | RESET | 写局奖励→env.reset |
| RESET | 重置完成 | COLLECT | — |
| COLLECT | step%checkpoint_interval==0 | COLLECT（自环） | checkpoint.save |
| TRAINING | global_step ≥ total_steps | DONE | 返回统计 |

自检：无死状态、无不可达状态；每条转移对应 Day06 一条测试用例。

## 7. 方法级设计表（核心方法）

| 方法 | 签名 | 前置条件 | 后置条件 | 设计落点 |
| --- | --- | --- | --- | --- |
| reset | reset(seed)->(obs,infos) | 配置合法 | obs 每项形状 (32,)、同种子逐元素一致 | env：重建 rng、随机站位、球回中 |
| step | step(actions)->五元组 | 未终局 | 步数+1；进球则全体 term | env：_apply_actions→_physics→_check_goal→_rewards→_observe |
| _apply_actions | (actions)->None | 动作索引合法 | 越界索引抛 ValueError（带索引值） | env：动作 0—7 移动、8 踢、9 停 |
| _check_goal | (ball)->int | 物理已推进 | 返回 0/1/2（无/蓝/红进球） | env：门区判定 |
| select_action | (agent_id, obs)->int | learner 仅收蓝队 id | 返回 ∈ [0,10) | 各算法 + TeamPolicy 分发 |
| update | (batch)->dict | batch 非空 | 损失有限；参数改变 | 各算法：批转张量→损失→反向→软更新 |
| HeuristicOpponent.select_action | (agent_id, obs)->int | — | 纯函数（同 obs 同输出） | training 层 |
| TeamPolicy.update | (batch)->dict | — | 仅蓝队转移进入 learner | training 层 |
| Evaluator.evaluate | (env,algo,n)->dict | — | 返回 win_rate/goal_diff/team_reward/inference_ms_p50 | training 层：真实计数与计时 |

## 8. 编码规范（10 条尺子，Day06/评审用）

1. 命名体现含义（动作→_apply_actions，不用 do1）2. 单方法 ≤40 行 3. 一类一责（物理归 env、张量归算法）4. 无未解释字面常量（全部经 config 或模块常量+注释）5. 无 print 调试输出（日志走 MetricLogger/print 仅限 scripts 入口）6. 无裸 except / 不吞异常 7. 不返回内部可变引用（观测每次新建数组）8. 布尔方法 is_xxx 9. 注释写"为什么" 10. 提交信息带类型前缀。

## 9. 附录 A：AI 使用与核对 / 附录 B：假设处置

本设计由 AI 起草，人工逐条核对：①所有维度（32/10/126/96）经"表内相加复核"✓ ②无幻觉组件（每个类都对应 HLD 类图）✓ ③公式与 MADDPG[9]/QMIX[10] 原文一致✓ ④术语与 SRS/HLD 统一✓ ⑤状态机无死状态✓。

假设处置：Q8 Gumbel-Softmax（采纳，待导师最终确认）；Q9 拼接状态（采纳，同上）；Q10 物理参数（采纳默认值，YAML 可调）；新增假设——"红队用规则对手即可提供有效对抗压力"（待 Day05 训练曲线观察后复核）。

---

*文档结束。评审通过后 tag v0.3-lld。*
