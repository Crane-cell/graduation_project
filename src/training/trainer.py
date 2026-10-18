# -*- coding: utf-8 -*-
"""训练主循环（F09）：采集（collect）与更新（update）双循环的调度者。

Day03 定稿主循环结构（真实结构）；env/algo 组件骨架期空转即可端到端跑通，
Day05 将组件替换为真实现——主循环本身不再改动。
对应教案"主循环归口一个调度者"：env 与 algo 均由外部构造注入（依赖注入，
保证可测试性——与 tetris 的 Random 构造传入同一原则）。
"""
from __future__ import annotations

import numpy as np

from ..algorithm.replay_buffer import Transition


class Trainer:
    def __init__(self, env, algo, buffer, logger, config: dict, checkpoint_manager=None) -> None:
        self.env = env
        self.algo = algo
        self.buffer = buffer
        self.logger = logger
        self.config = config
        self.checkpoint_manager = checkpoint_manager

        algo_cfg = config["algorithm"]
        train_cfg = config["train"]
        self.total_steps = int(train_cfg["total_steps"])
        self.eval_interval = int(train_cfg["eval_interval"])
        self.checkpoint_interval = int(train_cfg["checkpoint_interval"])
        self.batch_size = int(algo_cfg["batch_size"])
        self.update_start = int(algo_cfg["update_start"])
        self.update_every = int(algo_cfg.get("update_every", 4))

        self._global_step = 0

    def train(self) -> dict:
        """主循环：观测→决策→推进→存储→（达阈值后）更新→局末重置→周期存档。"""
        obs, _ = self.env.reset()
        episode_reward = 0.0
        while self._global_step < self.total_steps:
            # --- 采集（collect）---
            actions = self.algo.select_actions_batch(obs)
            next_obs, rewards, terms, truncs, _ = self.env.step(actions)
            self.buffer.add(
                Transition(obs=obs, actions=actions, rewards=rewards,
                           next_obs=next_obs, terminated=terms)
            )
            step_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            episode_reward += step_reward
            self.logger.write("train/step_reward", step_reward, self._global_step)

            obs = next_obs
            self._global_step += 1

            # --- 更新（update）：buffer 攒够阈值后按频率触发 ---
            if len(self.buffer) >= self.update_start and self._global_step % self.update_every == 0:
                losses = self.algo.update(self.buffer.sample(self.batch_size))
                for key, value in losses.items():
                    self.logger.write(f"train/loss_{key}", float(value), self._global_step)

            # --- 局末重置（episode 间循环）---
            episode_done = all(terms.values()) or all(truncs.values()) or not self.env.agents
            if episode_done:
                self.logger.write("train/episode_reward", episode_reward, self._global_step)
                episode_reward = 0.0
                obs, _ = self.env.reset()

            # --- 周期存档（F10）---
            if self.checkpoint_manager is not None and self._global_step % self.checkpoint_interval == 0:
                self.checkpoint_manager.save("latest", {"global_step": self._global_step})

        return {"global_step": self._global_step, "buffer_size": len(self.buffer)}
