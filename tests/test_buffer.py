# -*- coding: utf-8 -*-
"""经验回放测试（F09 组件）：容量淘汰、采样规模、固定 rng 可复现。"""
import random

import pytest

from src.algorithm.replay_buffer import ReplayBuffer, Transition


def make_transition(i: int) -> Transition:
    return Transition(
        obs={"blue_0": i}, actions={"blue_0": i}, rewards={"blue_0": float(i)},
        next_obs={"blue_0": i}, terminated={"blue_0": False},
    )


def test_capacity_eviction_keeps_latest():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.add(make_transition(i))
    assert len(buf) == 3
    sampled = buf.sample(3, rng=random.Random(0))
    values = {t.obs["blue_0"] for t in sampled}
    assert values == {2, 3, 4}  # 保留最新三条


@pytest.mark.parametrize("n_added,batch,expected", [(0, 4, 0), (3, 8, 3), (10, 4, 4)])
def test_sample_size_min(n_added, batch, expected):
    buf = ReplayBuffer(capacity=100)
    for i in range(n_added):
        buf.add(make_transition(i))
    assert len(buf.sample(batch, rng=random.Random(0))) == expected


def test_sample_fixed_rng_reproducible():
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        buf.add(make_transition(i))
    s1 = buf.sample(5, rng=random.Random(7))
    s2 = buf.sample(5, rng=random.Random(7))
    assert [t.obs["blue_0"] for t in s1] == [t.obs["blue_0"] for t in s2]
