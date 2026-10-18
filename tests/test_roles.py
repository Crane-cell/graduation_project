# -*- coding: utf-8 -*-
"""角色分工测试（F11）：轮转分配与独热编码。"""
import numpy as np
import pytest

from src.agent.roles import Role, assign_roles, role_onehot


@pytest.mark.parametrize("n,expected_prefix", [(1, [0]), (3, [0, 1, 2]), (6, [0, 1, 2, 0, 1, 2])])
def test_assign_roles_cycles(n, expected_prefix):
    """默认按 前锋→中场→后卫 循环。"""
    roles = assign_roles(n)
    assert len(roles) == n
    assert [int(r) for r in roles[: len(expected_prefix)]] == expected_prefix


@pytest.mark.parametrize("role", [Role.FORWARD, Role.MIDFIELDER, Role.DEFENDER])
def test_role_onehot(role):
    v = role_onehot(role)
    assert v.shape == (3,)
    assert v.sum() == pytest.approx(1.0)
    assert int(v[int(role)]) == 1
