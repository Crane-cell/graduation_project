# -*- coding: utf-8 -*-
"""IQL 基线（F12）：各智能体独立 DQN——把队友与对手都视为环境的一部分。

对照基线：验证 MARL 协作增益（NF-03：胜率提升 ≥20pp 以 IQL 为参照）。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAlgo
from .networks import batch_mlp_forward, batch_to_tensors, mlp, soft_update


class IQL(BaseAlgo):
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, config: dict) -> None:
        super().__init__(num_agents, obs_dim, action_dim, config)
        alg_cfg = config["algorithm"]
        hidden = int(alg_cfg.get("hidden_dim", 64))
        lr = float(alg_cfg.get("lr", 1e-3))
        self.gamma = float(alg_cfg.get("gamma", 0.99))
        self.tau_soft = float(alg_cfg.get("tau_soft", 0.01))
        self.epsilon = float(alg_cfg.get("epsilon", 0.1))

        n, o, a = num_agents, obs_dim, action_dim
        self.qs = nn.ModuleList([mlp(o, [hidden, hidden], a) for _ in range(n)])
        self.target_qs = deepcopy(self.qs)
        for p in self.target_qs.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.qs.parameters(), lr=lr)

    def select_action(self, agent_id: str, obs) -> int:
        i = int(agent_id.rsplit("_", 1)[1])
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            q = self.qs[i](torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def select_actions_batch(self, obs: dict) -> dict:
        """整队批量决策（Day08 优化）：一次张量化 + 向量化 ε-greedy + 一次 argmax。"""
        ids = list(obs)
        n = len(ids)
        o = torch.as_tensor(np.stack([np.asarray(obs[a], dtype=np.float32) for a in ids]))
        rnd = np.random.random(n) < self.epsilon
        acts = np.random.randint(self.action_dim, size=n)
        with torch.inference_mode():
            q = batch_mlp_forward(o, [self.qs[i] for i in range(n)])
            argmax = q.argmax(dim=1).numpy()
        acts[~rnd] = argmax[~rnd]
        return {a: int(v) for a, v in zip(ids, acts)}

    def update(self, batch: list) -> dict[str, float]:
        if not batch:
            return {}
        o, act, rew, o2, done = batch_to_tensors(batch)
        n = self.num_agents
        loss_sum = torch.zeros(())
        for i in range(n):
            q_taken = self.qs[i](o[:, i]).gather(1, act[:, i].unsqueeze(1)).squeeze(1)  # (B,)
            with torch.no_grad():
                q_next = self.target_qs[i](o2[:, i]).max(dim=1).values
                y = rew[:, i] + self.gamma * (1.0 - done) * q_next
            loss_sum = loss_sum + F.mse_loss(q_taken, y)
        loss = loss_sum / n
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        soft_update(self.qs, self.target_qs, self.tau_soft)
        return {"td_loss": float(loss.item())}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"qs": self.qs.state_dict(), "target_qs": self.target_qs.state_dict()}, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.qs.load_state_dict(state["qs"])
        self.target_qs.load_state_dict(state["target_qs"])
