# -*- coding: utf-8 -*-
"""共用网络组件（F06）：MLP 构造、Gumbel-Softmax（ST）、批转张量、软更新。

本模块只依赖 torch 与 numpy——algorithm 层不认识任何环境类型（HLD 红线）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    """标准 MLP：隐藏层 ReLU，输出层线性。"""
    modules: list[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        modules.append(nn.Linear(d, h))
        modules.append(nn.ReLU())
        d = h
    modules.append(nn.Linear(d, out_dim))
    return nn.Sequential(*modules)


def batch_mlp_forward(x: torch.Tensor, modules: nn.ModuleList) -> torch.Tensor:
    """n 个同构 MLP 的整队批量前向（Day08 优化核心）。

    把 n 个 MLP 的逐层权重堆叠成 (n, out, in)，一次 bmm 算一整层，
    消除逐智能体模块分发的 Python 开销（cProfile 实测约占决策耗时 70%）。
    x: (n, in) → (n, out)；modules: 长度 n 的同构 nn.Sequential。
    数学结果与逐个调用一致（同一份权重），仅浮点运算次序略有差异。
    """
    xb = x.unsqueeze(1)  # (n, 1, in)
    n = len(modules)
    for li, layer in enumerate(modules[0]):
        if isinstance(layer, nn.Linear):
            w = torch.stack([modules[i][li].weight for i in range(n)])  # (n, out, in)
            b = torch.stack([modules[i][li].bias for i in range(n)])    # (n, out)
            xb = torch.bmm(xb, w.transpose(-1, -2)) + b.unsqueeze(1)    # (n, 1, out)
        else:  # ReLU 等逐元素激活
            xb = layer(xb)
    return xb.squeeze(1)


def gumbel_softmax_hard(logits: torch.Tensor, tau: float = 1.0, eps: float = 1e-20) -> torch.Tensor:
    """Gumbel-Softmax with Straight-Through：前向 one-hot，反向用软分布梯度（Q8 方案）。"""
    u = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(u + eps) + eps)
    y_soft = F.softmax((logits + gumbel) / tau, dim=-1)
    idx = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
    return (y_hard - y_soft).detach() + y_soft


def batch_to_tensors(batch: list, device: str = "cpu"):
    """list[Transition]（dict 字段）→ (obs, act, rew, next_obs, done) 五个张量。

    obs: (B, n, obs_dim) float32；act: (B, n) int64；rew: (B, n) float32；
    next_obs 同 obs；done: (B,) float32（全队同终止，取首个智能体）。
    智能体顺序按编号排序，保证确定性。
    """
    from .replay_buffer import Transition  # 局部 import 避免循环（replay_buffer 仅依赖 dataclass）

    agents = sorted(batch[0].obs.keys(), key=lambda s: int(s.rsplit("_", 1)[1]))
    n = len(agents)
    batch_size = len(batch)

    obs = torch.zeros((batch_size, n, len(np.asarray(batch[0].obs[agents[0]]))), dtype=torch.float32)
    act = torch.zeros((batch_size, n), dtype=torch.int64)
    rew = torch.zeros((batch_size, n), dtype=torch.float32)
    next_obs = torch.zeros_like(obs)
    done = torch.zeros(batch_size, dtype=torch.float32)

    for b, tr in enumerate(batch):
        for i, a in enumerate(agents):
            obs[b, i] = torch.as_tensor(np.asarray(tr.obs[a]), dtype=torch.float32)
            act[b, i] = int(tr.actions[a])
            rew[b, i] = float(tr.rewards[a])
            next_obs[b, i] = torch.as_tensor(np.asarray(tr.next_obs[a]), dtype=torch.float32)
        done[b] = float(tr.terminated[agents[0]])

    if device != "cpu":
        obs, act, rew, next_obs, done = (
            obs.to(device), act.to(device), rew.to(device), next_obs.to(device), done.to(device)
        )
    return obs, act, rew, next_obs, done


def soft_update(source: nn.ModuleList, target: nn.ModuleList, tau: float) -> None:
    """目标网络软更新：θ' ← τθ + (1-τ)θ'。"""
    with torch.no_grad():
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.mul_(1.0 - tau)
            tp.add_(tau * sp)
