# -*- coding: utf-8 -*-
"""QMIX（F08，Q9 方案）：个体 Q 网络 + 单调混合网络。

混合网络以"全体智能体观测拼接"为全局状态 s_t（96 维 @3v3），
权重经超网络生成并取绝对值保证非负（单调性）；TD 目标用目标网络。
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


class QMixer(nn.Module):
    """单调混合网络：Q_tot = w2^T · ELU(w1^T q + b1) + b2，w1/w2 非负。"""

    def __init__(self, state_dim: int, embed_dim: int, n_agents: int) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = mlp(state_dim, [embed_dim], n_agents * embed_dim)
        self.hyper_b1 = mlp(state_dim, [embed_dim], embed_dim)
        self.hyper_w2 = mlp(state_dim, [embed_dim], embed_dim)
        self.hyper_b2 = mlp(state_dim, [embed_dim], 1)

    def forward(self, state: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """state: (B, S)；q: (B, n)（各智能体所取动作的 Q 值）→ Q_tot: (B, 1)。"""
        b = state.size(0)
        w1 = self.hyper_w1(state).abs().view(b, self.n_agents, self.embed_dim)  # (B, n, e)
        b1 = self.hyper_b1(state)  # (B, e)
        # hidden_j = Σ_i q_i · w1[i, j]：q (B,1,n) × w1 (B,n,e) → (B,1,e)
        hidden = F.elu(torch.bmm(q.reshape(b, 1, self.n_agents), w1).squeeze(1) + b1)  # (B, e)
        w2 = self.hyper_w2(state).abs()  # (B, e)
        b2 = self.hyper_b2(state)  # (B, 1)
        return torch.bmm(w2.unsqueeze(1), hidden.unsqueeze(2)).squeeze(2) + b2  # (B, 1)


class QMIX(BaseAlgo):
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, config: dict) -> None:
        super().__init__(num_agents, obs_dim, action_dim, config)
        alg_cfg = config["algorithm"]
        hidden = int(alg_cfg.get("hidden_dim", 64))
        embed = int(alg_cfg.get("embed_dim", 32))
        lr = float(alg_cfg.get("lr", 1e-3))
        self.gamma = float(alg_cfg.get("gamma", 0.99))
        self.tau_soft = float(alg_cfg.get("tau_soft", 0.01))
        self.epsilon = float(alg_cfg.get("epsilon", 0.1))

        n, o, a = num_agents, obs_dim, action_dim
        self.qs = nn.ModuleList([mlp(o, [hidden, hidden], a) for _ in range(n)])
        self.mixer = QMixer(n * o, embed, n)
        self.target_qs = deepcopy(self.qs)
        self.target_mixer = deepcopy(self.mixer)
        for p in self.target_qs.parameters():
            p.requires_grad_(False)
        for p in self.target_mixer.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(list(self.qs.parameters()) + list(self.mixer.parameters()), lr=lr)

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
        batch_size = o.size(0)

        state, state2 = o.reshape(batch_size, -1), o2.reshape(batch_size, -1)

        q_taken = torch.stack(
            [self.qs[i](o[:, i]).gather(1, act[:, i].unsqueeze(1)).squeeze(1) for i in range(n)], dim=1
        )  # (B, n)
        q_tot = self.mixer(state, q_taken).squeeze(-1)  # (B,)

        with torch.no_grad():
            # 单调性保证：个体 argmax 即联合最优
            tgt_acts = torch.stack(
                [self.target_qs[i](o2[:, i]).argmax(dim=1) for i in range(n)], dim=1
            )  # (B, n)
            q2_taken = torch.stack(
                [
                    self.target_qs[i](o2[:, i]).gather(1, tgt_acts[:, i].unsqueeze(1)).squeeze(1)
                    for i in range(n)
                ],
                dim=1,
            )
            q2_tot = self.target_mixer(state2, q2_taken).squeeze(-1)
            r_team = rew.sum(dim=1)
            y = r_team + self.gamma * (1.0 - done) * q2_tot

        loss = F.mse_loss(q_tot, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        soft_update(self.qs, self.target_qs, self.tau_soft)
        soft_update(self.mixer, self.target_mixer, self.tau_soft)

        return {"td_loss": float(loss.item())}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "qs": self.qs.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_qs": self.target_qs.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.qs.load_state_dict(state["qs"])
        self.mixer.load_state_dict(state["mixer"])
        self.target_qs.load_state_dict(state["target_qs"])
        self.target_mixer.load_state_dict(state["target_mixer"])
