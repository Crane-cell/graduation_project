# -*- coding: utf-8 -*-
"""环境层单元测试（F01—F04）：物理、观测、奖励、终局、确定性。

覆盖 LLD 状态机中环境侧全部分支：动作 0—7/8/9、进球两侧、非门区贴墙、
上下墙反弹、智能体碰撞推开、球-智能体碰撞、步数上限终局、非法动作。
"""
import numpy as np
import pytest

from src.env.soccer_env import SoccerEnv


def all_stay(env) -> dict:
    return {a: 9 for a in env.possible_agents}


class TestResetAndObserve:
    def test_reset_same_seed_deterministic(self, env_cfg):
        """NF-02：同种子两次 reset，观测逐元素一致。"""
        e1, e2 = SoccerEnv(env_cfg), SoccerEnv(env_cfg)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        for a in o1:
            assert np.array_equal(o1[a], o2[a])

    def test_reset_different_seed_differs(self, env_cfg):
        e = SoccerEnv(env_cfg)
        o1, _ = e.reset(seed=1)
        o2, _ = e.reset(seed=2)
        assert any(not np.array_equal(o1[a], o2[a]) for a in o1)

    def test_obs_shape_and_range(self, env_cfg):
        """SRS GWT-2：形状 (32,)，取值 [-1,1]。"""
        env = SoccerEnv(env_cfg)
        obs, _ = env.reset(seed=0)
        for a, o in obs.items():
            assert o.shape == (32,)
            assert float(o.min()) >= -1.0 and float(o.max()) <= 1.0001  # 浮点余量

    def test_trajectory_reproducible_given_actions(self, env_cfg):
        """NF-02：同种子同动作序列 → 逐步观测一致（10 步）。"""
        e1, e2 = SoccerEnv(env_cfg), SoccerEnv(env_cfg)
        o1, _ = e1.reset(seed=7)
        o2, _ = e2.reset(seed=7)
        for _ in range(10):
            acts = {a: int(np.random.default_rng(3).integers(10)) for a in e1.possible_agents}
            r1 = e1.step(acts)
            r2 = e2.step(acts)
            for a in e1.possible_agents:
                assert np.array_equal(r1[0][a], r2[0][a])
                assert r1[1][a] == pytest.approx(r2[1][a])


class TestActions:
    @pytest.mark.parametrize("bad", [-1, 10, 3.5, "x", None])
    def test_invalid_action_raises(self, env_cfg, bad):
        """SRS GWT-3：非法动作索引抛出明确异常（非静默失败）。"""
        env = SoccerEnv(env_cfg)
        env.reset(seed=0)
        acts = all_stay(env)
        acts["blue_0"] = bad
        with pytest.raises(ValueError):
            env.step(acts)

    def test_move_east_increases_x(self, env_cfg):
        env = SoccerEnv(env_cfg)
        env.reset(seed=0)
        x_before = env._pos[0][0]
        acts = all_stay(env)
        acts["blue_0"] = 0  # 东
        env.step(acts)
        assert env._pos[0][0] - x_before >= 0.8

    def test_kick_without_control_noop(self, env_cfg):
        """远离球时踢球无效果（LLD 动作 8 语义）。"""
        env = SoccerEnv(env_cfg)
        env.reset(seed=0)
        env._pos[0] = [2.0, 2.0]   # 远离中心球
        env._ball = np.array([10.0, 7.5])
        env._ball_vel = np.array([0.0, 0.0])
        acts = all_stay(env)
        acts["blue_0"] = 8
        env.step(acts)
        assert float(np.linalg.norm(env._ball_vel)) == pytest.approx(0.0, abs=1e-9)

    def test_kick_with_control_sets_ball_velocity(self, env_cfg):
        """控球踢球：本步按踢出速度 3.0 位移，步末摩擦 ×0.9 折成 2.7（LLD 物理：先积分后摩擦）。"""
        env = SoccerEnv(env_cfg)
        env.reset(seed=0)
        env._pos[0] = [9.6, 7.5]
        env._heading[0] = [1.0, 0.0]
        env._ball = np.array([10.0, 7.5])
        env._ball_vel = np.array([0.0, 0.0])
        acts = all_stay(env)
        acts["blue_0"] = 8
        env.step(acts)
        assert float(env._ball[0]) == pytest.approx(13.0)   # 本步位移用踢出速度 3.0
        assert env._ball_vel[0] == pytest.approx(2.7)        # 步末摩擦 3.0 × 0.9
        assert abs(env._ball_vel[1]) < 1e-9                 # 本布局无碰撞，垂直分量为零


class TestPhysics:
    def test_agent_overlap_separated(self, env_cfg):
        """智能体重叠 → 连线推开至 ≥2r。"""
        env = SoccerEnv(env_cfg)
        env.reset(seed=0)
        env._pos[0] = [5.0, 5.0]
        env._pos[1] = [5.2, 5.0]
        env.step(all_stay(env))
        dist = float(np.linalg.norm(env._pos[0] - env._pos[1]))
        assert dist >= 0.59

    def test_ball_agent_collision_pushed_out(self, env_cfg):
        """球距智能体过近 → 推出到 ≥ r_a+r_b。"""
        env = SoccerEnv(env_cfg)
        env.reset(seed=0)
        env._pos[0] = [5.0, 5.0]
        env._ball = np.array([5.1, 5.0])
        env._ball_vel = np.array([0.0, 0.0])
        env.step(all_stay(env))
        dist = float(np.linalg.norm(env._ball - env._pos[0]))
        assert dist >= 0.44

    def test_ball_top_wall_reflect(self, env_cfg):
        env = SoccerEnv(env_cfg)
        env.reset(seed=0)
        env._ball = np.array([10.0, 14.9])
        env._ball_vel = np.array([0.0, 2.0])
        env.step(all_stay(env))
        assert env._ball[1] <= 15.0 - 0.15 + 1e-6
        assert env._ball_vel[1] < 0


class TestGoalAndTermination:
    @pytest.mark.parametrize(
        "ball,vel,expect_goal",
        [
            ([19.5, 7.5], [1.0, 0.0], 1),    # 蓝方进球（右门内）
            ([0.5, 7.5], [-1.0, 0.0], 2),    # 红方进球（左门内）
            ([19.5, 1.0], [1.0, 0.0], 0),     # 右侧非门区 → 贴墙
            ([1.0, 14.0], [0.0, 3.0], 0),    # 上墙反弹
        ],
    )
    def test_goal_detection_parametrized(self, env_cfg, ball, vel, expect_goal):
        """门区判定 + 围墙分支（SRS GWT-1 邻域）。"""
        env = SoccerEnv(env_cfg)
        env.reset(seed=0)
        env._ball = np.array(ball, dtype=float)
        env._ball_vel = np.array(vel, dtype=float)
        _obs, _rew, terms, _trunc, infos = env.step(all_stay(env))
        assert infos["blue_0"]["goal"] == expect_goal
        assert all(terms.values()) == (expect_goal != 0)

    def test_goal_rewards_signs(self, env_cfg):
        """SRS GWT-4：蓝方进球 → 蓝队奖励含 +goal、红队含 -goal。"""
        env = SoccerEnv(env_cfg)
        env.reset(seed=0)
        env._ball = np.array([19.5, 7.5])
        env._ball_vel = np.array([1.0, 0.0])
        _o, rew, _t, _tr, _i = env.step(all_stay(env))
        assert rew["blue_0"] >= 10.0 - 1e-6
        assert rew["red_0"] <= -10.0 + 1e-6

    def test_step_limit_terminates(self, env_cfg):
        """步数上限终局（防不进球死循环，Q10）。"""
        env_cfg["env"]["max_episode_steps"] = 3
        env = SoccerEnv(env_cfg)
        env.reset(seed=0)
        for _ in range(2):
            env.step(all_stay(env))
        assert env.agents  # 未结束
        _o, _r, terms, _tr, _i = env.step(all_stay(env))
        assert all(terms.values())
        assert env.agents == []
        with pytest.raises(RuntimeError):
            env.step(all_stay(env))

    def test_step_after_done_raises(self, env_cfg):
        env = SoccerEnv(env_cfg)
        env.reset(seed=0)
        env.step(all_stay(env))
        env.step(all_stay(env))
        env._step_count = env.max_episode_steps  # 触发终局
        env.step(all_stay(env))
        with pytest.raises(RuntimeError):
            env.step(all_stay(env))

    def test_control_bonus_reward(self, env_cfg):
        """控球塑形：距离 0.47（<0.5 控球且 >0.45 不触发碰撞）。"""
        env = SoccerEnv(env_cfg)
        env.reset(seed=0)
        env._pos[0] = [9.53, 7.5]
        env._ball = np.array([10.0, 7.5])
        env._ball_vel = np.array([0.0, 0.0])
        _o, rew, _t, _tr, _i = env.step(all_stay(env))
        assert rew["blue_0"] >= 0.02


class TestSpaces:
    def test_spaces_match_config(self, env_cfg):
        env = SoccerEnv(env_cfg)
        assert env.observation_space("blue_0").shape == (32,)
        assert env.action_space("blue_0").n == 10
