# -*- coding: utf-8 -*-
"""检查点存取（F10，Day07 数据驱动扩展）：保存 / 恢复 / 最新查找 + 版本兼容。

设计要点：
  - 每次保存写入 schema_version（结构演进可识别），saved_at（落盘时间戳）
  - 读取时：旧档（无 schema_version，记 0）自动兜底并告警，兼容继续训练
  - 版本高于当前支持 → 显式报错（防静默误读未来格式）
"""
import time
import warnings
from pathlib import Path

import torch

SCHEMA_VERSION = 1  # 检查点结构版本（结构变化时递增，旧档自动兼容）


class CheckpointManager:
    def __init__(self, save_dir: str | Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, state: dict) -> str:
        """保存（附版本号与落盘时间戳；state 内同名字段以元信息为准）。"""
        path = self.save_dir / f"{name}.pt"
        payload = {
            **state,
            "schema_version": SCHEMA_VERSION,
            "saved_at": time.time(),
            "global_step": state.get("global_step", 0),
        }
        torch.save(payload, path)
        return str(path)

    def load(self, name: str) -> dict:
        """恢复检查点；旧版（无 schema_version）自动按 v1 兼容读取并告警。"""
        path = self.save_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"检查点不存在：{path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise ValueError(f"检查点格式异常（非字典）：{path}")
        version = int(data.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"检查点 {name} 版本 v{version} 高于当前支持 v{SCHEMA_VERSION}，请升级代码后读取"
            )
        if version == 0:
            warnings.warn(
                f"检查点 {name} 为旧格式（无 schema_version），已按 v{SCHEMA_VERSION} 兼容读取",
                UserWarning,
                stacklevel=2,
            )
            data.setdefault("schema_version", SCHEMA_VERSION)
            data.setdefault("saved_at", 0.0)
            data.setdefault("global_step", 0)
        return data

    def latest(self) -> str | None:
        """最新检查点名（按文件修改时间），无检查点返回 None。"""
        cands = sorted(self.save_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        return cands[-1].stem if cands else None