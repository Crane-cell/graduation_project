# -*- coding: utf-8 -*-
"""算法层测试（F06—F08/F12）：前向形状、更新有限性与参数变化、Gumbel-ST、软更新。

可测试性三原则落地：固定种子、逻辑与 IO 分离、相同输入相同输出。
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.algorithm.iql import IQL
from src.algorithm.maddpg import MADDPG
from src.algorithm.networks import batch_to_tensors, gumbel_softmax_hard, mlp, soft_update
from src.algorithm.qmix import QMIX, QMixer


def snapshot_params(module) -> list:
    return [p.detach().clone() for p in module.parameters()]


def assert_params_changed(before, after):
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


class TestMADDPG:
    @pytest.mark.parametrize("agent_idx", [0, 1, 2])
    def test_select_action_range_deterministic(self, algo_cfg, agent_idx):
        """ε=0 → argmax 确定性决策，动作 ∈ [0, A)。"""
        algo = MADDPG(num_agents=3, obs_dim=8, action_dim=4, config=algo_cfg)
        obs = np.zeros(8, dtype=np.float32)
        a1 = algo.select_action(f"blue_{agent_idx}", obs)
        a2 = algo.select_action(f"blue_{agent_idx}", obs)
        assert a1 == a2
        assert 0 <= a1 < 4

    def test_update_finite_and_changes_params(self, algo_cfg, make_batch):
        """SRS GWT-7：损失有限；演员与评论家参数均变化。"""
        algo = MADDPG(num_agents=3, obs_dim=8, action_dim=4, config=algo_cfg)
        actors_before = snapshot_params(algo.actors)
        critics_before = snapshot_params(algo.critics)
        losses = algo.update(make_batch(batch_size=8))
        assert np.isfinite(losses["critic_loss"]) and np.isfinite(losses["actor_loss"])
        assert_params_changed(actors_before, snapshot_params(algo.actors))
        assert_params_changed(critics_before, snapshot_params(algo.critics))

    def test_empty_batch_returns_empty(self, algo_cfg):
        algo = MADDPG(num_agents=3, obs_dim=8, action_dim=4, config=algo_cfg)
        assert algo.update([]) == {}


class TestQMIX:
    def test_mixer_forward_shape(self, algo_cfg):
        mixer = QMixer(state_dim=3 * 8, embed_dim=8, n_agents=3)
        state = torch.randn(5, 24)
        q = torch.randn(5, 3)
        out = mixer(state, q)
        assert out.shape == (5, 1)
        assert torch.isfinite(out).all()

    def test_update_finite_and_changes_params(self, algo_cfg, make_batch):
        algo = QMIX(num_agents=3, obs_dim=8, action_dim=4, config=algo_cfg)
        qs_before = snapshot_params(algo.qs)
        mixer_before = snapshot_params(algo.mixer)
        losses = algo.update(make_batch(batch_size=8))
        assert np.isfinite(losses["td_loss"])
        assert_params_changed(qs_before, snapshot_params(algo.qs))
        assert_params_changed(mixer_before, snapshot_params(algo.mixer))

    def test_save_load_roundtrip(self, algo_cfg, tmp_path):
        algo = QMIX(num_agents=3, obs_dim=8, action_dim=4, config=algo_cfg)
        path = str(tmp_path / "qmix.pt")
        algo.save(path)
        ref = algo.qs[0](torch.zeros(1, 8))
        algo.load(path)
        assert torch.equal(algo.qs[0](torch.zeros(1, 8)), ref)


class TestIQL:
    def test_update_finite_and_changes_params(self, algo_cfg, make_batch):
        algo = IQL(num_agents=3, obs_dim=8, action_dim=4, config=algo_cfg)
        before = snapshot_params(algo.qs)
        losses = algo.update(make_batch(batch_size=8))
        assert np.isfinite(losses["td_loss"])
        assert_params_changed(before, snapshot_params(algo.qs))


class TestBatchSelectConsistency:
    """Day08 性能契约回归：批量前向与逐智能体循环在 ε=0 时动作逐位一致。"""

    @pytest.mark.parametrize("cls", [MADDPG, QMIX, IQL])
    def test_batch_matches_loop_epsilon_zero(self, algo_cfg, cls):
        cfg = {**algo_cfg, "algorithm": {**algo_cfg["algorithm"], "epsilon": 0.0}}
        algo = cls(num_agents=3, obs_dim=8, action_dim=4, config=cfg)
        rng = np.random.default_rng(0)
        obs = {f"blue_{i}": rng.normal(size=8).astype(np.float32) for i in range(3)}
        loop = {a: algo.select_action(a, obs[a]) for a in obs}
        batch = algo.select_actions_batch(obs)
        assert batch == loop

    @pytest.mark.parametrize("cls", [MADDPG, QMIX, IQL])
    def test_batch_returns_dict_for_all_agents(self, algo_cfg, cls):
        algo = cls(num_agents=3, obs_dim=8, action_dim=4, config=algo_cfg)
        rng = np.random.default_rng(1)
        obs = {f"blue_{i}": rng.normal(size=8).astype(np.float32) for i in range(3)}
        out = algo.select_actions_batch(obs)
        assert set(out) == set(obs)
        assert all(0 <= v < 4 for v in out.values())

    def test_batch_mlp_forward_matches_loop(self):
        """批量前向与逐智能体前向数值一致（Day08 优化等价性）。"""
        from src.algorithm.networks import batch_mlp_forward

        nets = nn.ModuleList([mlp(8, [16, 16], 4) for _ in range(3)])
        x = torch.randn(3, 8)
        batched = batch_mlp_forward(x, nets)
        loop = torch.stack([nets[i](x[i : i + 1]).squeeze(0) for i in range(3)])
        assert batched.shape == (3, 4)
        assert torch.allclose(batched, loop, atol=1e-5)

    def test_batch_mlp_forward_shared_weights(self):
        """批量前向使用同一份权重（与逐个调用结果逐层一致）。"""
        from src.algorithm.networks import batch_mlp_forward

        nets = nn.ModuleList([mlp(4, [8], 2) for _ in range(2)])
        x = torch.randn(2, 4)
        out = batch_mlp_forward(x, nets)
        single = torch.stack([nets[i](x[i : i + 1]).squeeze(0) for i in range(2)])
        assert torch.allclose(out, single, atol=1e-5)


class TestNetworks:
    def test_mlp_output_shape(self):
        net = mlp(8, [16, 16], 4)
        assert net(torch.randn(7, 8)).shape == (7, 4)

    def test_gumbel_softmax_hard_is_onehot(self):
        """前向为 one-hot（ST），反向对 logits 可导。"""
        logits = torch.randn(6, 4, requires_grad=True)
        out = gumbel_softmax_hard(logits, tau=1.0)
        assert out.shape == (6, 4)
        assert torch.allclose(out.sum(dim=-1), torch.ones(6))
        assert ((out == 0.0) | (out == 1.0)).all()
        out.sum().backward()  # 梯度可回传
        assert logits.grad is not None

    def test_batch_to_tensors_shapes(self, make_batch):
        batch = make_batch(n_agents=3, obs_dim=8, batch_size=6)
        o, a, r, o2, d = batch_to_tensors(batch)
        assert o.shape == (6, 3, 8) and o.dtype == torch.float32
        assert a.shape == (6, 3) and a.dtype == torch.int64
        assert r.shape == (6, 3) and r.dtype == torch.float32
        assert o2.shape == (6, 3, 8)
        assert d.shape == (6,)

    def test_soft_update_full(self):
        src, tgt = mlp(4, [8], 2), mlp(4, [8], 2)
        soft_update(src, tgt, tau=1.0)
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            assert torch.equal(sp, tp)

    def test_soft_update_partial(self):
        src, tgt = mlp(4, [8], 2), mlp(4, [8], 2)
        before = [p.detach().clone() for p in tgt.parameters()]
        soft_update(src, tgt, tau=0.0)
        for b, tp in zip(before, tgt.parameters()):
            assert torch.equal(b, tp)
