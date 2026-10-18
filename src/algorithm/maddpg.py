# -*- coding: utf-8 -*-
"""MADDPG（F07，Q8 方案）：集中式评论家 + 分散式演员 + Gumbel-Softmax 离散化。

网络结构见 LLD 第 5 章：
  actor_i  : MLP(obs_dim → 64 → 64 → action_dim)，输出 logits
  critic_i : MLP(n*obs_dim + n*action_dim → 64 → 64 → 1)
执行 ε-greedy；训练用 Gumbel-ST 把离散动作接入确定性策略梯度。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAlgo
from .networks import batch_mlp_forward, batch_to_tensors, gumbel_softmax_hard, mlp, soft_update


class MADDPG(BaseAlgo):
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, config: dict) -> None:
        super().__init__(num_agents, obs_dim, action_dim, config)
        alg_cfg = config["algorithm"]
        hidden = int(alg_cfg.get("hidden_dim", 64))
        lr = float(alg_cfg.get("lr", 1e-3))
        self.gamma = float(alg_cfg.get("gamma", 0.99))
        self.tau_soft = float(alg_cfg.get("tau_soft", 0.01))
        self.epsilon = float(alg_cfg.get("epsilon", 0.1))

        n, o, a = num_agents, obs_dim, action_dim
        self.actors = nn.ModuleList([mlp(o, [hidden, hidden], a) for _ in range(n)])
        self.critics = nn.ModuleList([mlp(n * o + n * a, [hidden, hidden], 1) for _ in range(n)])
        self.target_actors = deepcopy(self.actors)
        self.target_critics = deepcopy(self.critics)
        for p in self.target_actors.parameters():
            p.requires_grad_(False)
        for p in self.target_critics.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actors.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=lr)

    # ---------- 契约实现 ----------

    def select_action(self, agent_id: str, obs) -> int:
        i = int(agent_id.rsplit("_", 1)[1])
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            logits = self.actors[i](torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0))
        return int(logits.argmax(dim=1).item())

    def select_actions_batch(self, obs: dict) -> dict:
        """整队批量决策（Day08 优化）：一次张量化 + 向量化 ε-greedy + 一次 argmax。

        与逐智能体循环语义完全一致（同 ε 随机序列、同 argmax 结果），
        但省去 n 次 np.asarray/torch.as_tensor/argmax 的 Python 侧开销。
        """
        ids = list(obs)
        n = len(ids)
        o = torch.as_tensor(np.stack([np.asarray(obs[a], dtype=np.float32) for a in ids]))
        rnd = np.random.random(n) < self.epsilon
        acts = np.random.randint(self.action_dim, size=n)
        with torch.inference_mode():
            logits = batch_mlp_forward(o, [self.actors[i] for i in range(n)])
            argmax = logits.argmax(dim=1).numpy()
        acts[~rnd] = argmax[~rnd]
        return {a: int(v) for a, v in zip(ids, acts)}

    def update(self, batch: list) -> dict[str, float]:
        if not batch:
            return {}
        o, act, rew, o2, done = batch_to_tensors(batch)
        n = self.num_agents
        a_dim = self.action_dim
        batch_size = o.size(0)

        a_onehot = F.one_hot(act, a_dim).float()  # (B, n, A)
        critic_in = torch.cat([o.reshape(batch_size, -1), a_onehot.reshape(batch_size, -1)], dim=1)

        # --- 目标 Q 值（确定性目标策略：argmax one-hot）---
        with torch.no_grad():
            tgt_logits = torch.stack([self.target_actors[i](o2[:, i]) for i in range(n)], dim=1)
            tgt_onehot = F.one_hot(tgt_logits.argmax(dim=-1), a_dim).float()
            critic_in2 = torch.cat([o2.reshape(batch_size, -1), tgt_onehot.reshape(batch_size, -1)], dim=1)
            y = torch.stack(
                [
                    rew[:, i] + self.gamma * (1.0 - done) * self.target_critics[i](critic_in2).squeeze(-1)
                    for i in range(n)
                ],
                dim=1,
            )  # (B, n)

        # --- 评论家更新 ---
        q_all = torch.stack([self.critics[i](critic_in).squeeze(-1) for i in range(n)], dim=1)
        critic_loss = F.mse_loss(q_all, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- 演员更新：Gumbel-ST 伪动作仅对第 i 维可导 ---
        logits_all = torch.stack([self.actors[i](o[:, i]) for i in range(n)], dim=1)  # (B, n, A)
        gumbel = gumbel_softmax_hard(logits_all, tau=1.0)  # (B, n, A)
        actor_loss = torch.zeros(())
        for i in range(n):
            pieces = [gumbel[:, j] if j == i else a_onehot[:, j] for j in range(n)]
            cin = torch.cat([o.reshape(batch_size, -1)] + pieces, dim=1)
            actor_loss = actor_loss - self.critics[i](cin).squeeze(-1).mean()
        actor_loss = actor_loss / n
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.actors, self.target_actors, self.tau_soft)
        soft_update(self.critics, self.target_critics, self.tau_soft)

        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actors": self.actors.state_dict(),
                "critics": self.critics.state_dict(),
                "target_actors": self.target_actors.state_dict(),
                "target_critics": self.target_critics.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.actors.load_state_dict(state["actors"])
        self.critics.load_state_dict(state["critics"])
        self.target_actors.load_state_dict(state["target_actors"])
        self.target_critics.load_state_dict(state["target_critics"])
