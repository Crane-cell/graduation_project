# -*- coding: utf-8 -*-
"""配置加载与校验（F05，Day07 数据驱动扩展）：YAML → 校验 → 兜底。

三段式（改配置不动代码，坏配置不静默）：
  1. 读取  ：YAML → dict（顶层必须为映射）
  2. 校验  ：类型 / 数值范围 / 允许集合违例 → 抛 ValueError（含全部违例明细）
  3. 兜底  ：缺失字段用 DEFAULTS 补齐；未知字段显式警告（防拼写错误静默吞掉）
"""
from __future__ import annotations

import warnings
from pathlib import Path

import yaml

# ---------- 默认配置（与 src/configs/default.yaml 同源口径） ----------
DEFAULTS: dict = {
    "env": {
        "num_per_side": 3,
        "obs_dim": 32,
        "action_dim": 10,
        "max_episode_steps": 500,
        "goal_half_width": 2.0,
        "field": {"width": 20.0, "height": 15.0},
        "reward": {"goal": 10.0, "control": 0.02, "advance": 0.5, "approach": 0.005},
    },
    "algorithm": {
        "name": "maddpg",
        "gamma": 0.99,
        "lr": 0.001,
        "batch_size": 32,
        "buffer_size": 100000,
        "update_start": 1000,
        "update_every": 4,
        "hidden_dim": 64,
        "embed_dim": 32,
        "tau_soft": 0.01,
        "epsilon": 0.1,
        # 消融开关（开题 5.4 实验设计）：同一训练代码按开关剪枝算法组件
        "ablation": {
            "communication": False,       # 有限通信机制（预留，M3 后启用）
            "role_assignment": True,      # 角色分工（前锋/中场/后卫）
            "value_decomposition": True,  # 值函数分解（QMIX 关掉退化为独立 Q）
        },
    },
    "train": {
        "total_steps": 100000,
        "eval_interval": 10000,
        "checkpoint_interval": 5000,
        "seed": 42,
    },
    "eval": {"n_episodes": 1000, "seed": 2026},
    "logging": {"log_dir": "runs"},
}

# 字段规则：类型 + 约束（None=仅类型；set=允许值；二元组=闭区间）
_RULES: dict = {
    "env": {
        "num_per_side": (int, (1, 11)),
        "obs_dim": (int, (8, 1024)),
        "action_dim": (int, (2, 64)),
        "max_episode_steps": (int, (1, 1 << 20)),
        "goal_half_width": (float, (0.1, 10.0)),
    },
    "env.field": {
        "width": (float, (5.0, 200.0)),
        "height": (float, (5.0, 200.0)),
    },
    "env.reward": {
        "goal": (float, (-100.0, 100.0)),
        "control": (float, (-10.0, 10.0)),
        "advance": (float, (-10.0, 10.0)),
        "approach": (float, (-10.0, 10.0)),
    },
    "algorithm": {
        "name": (str, {"maddpg", "qmix", "iql"}),
        "gamma": (float, (0.0, 1.0)),
        "lr": (float, (1e-8, 1.0)),
        "batch_size": (int, (1, 1 << 16)),
        "buffer_size": (int, (1, 1 << 24)),
        "update_start": (int, (0, 1 << 30)),
        "update_every": (int, (1, 1 << 16)),
        "hidden_dim": (int, (4, 4096)),
        "embed_dim": (int, (4, 4096)),
        "tau_soft": (float, (0.0, 1.0)),
        "epsilon": (float, (0.0, 1.0)),
    },
    "algorithm.ablation": {
        "communication": (bool, None),
        "role_assignment": (bool, None),
        "value_decomposition": (bool, None),
    },
    "train": {
        "total_steps": (int, (1, 1 << 40)),
        "eval_interval": (int, (1, 1 << 40)),
        "checkpoint_interval": (int, (1, 1 << 40)),
        "seed": (int, None),
    },
    "eval": {"n_episodes": (int, (1, 1 << 20)), "seed": (int, None)},
    "logging": {"log_dir": (str, None)},
}


def _coerce(name: str, value, rule) -> object:
    """按规则做类型强转（int 接受整值 float；float 接受 int）与范围/集合校验。"""
    typ, bounds = rule
    if isinstance(value, bool) and typ is not bool:
        raise ValueError(f"字段 {name!r} 不能是布尔值（{value!r}）")
    if typ is bool:
        if isinstance(value, bool):
            ok_value = value
        elif isinstance(value, int) and value in (0, 1):
            ok_value = bool(value)  # YAML 的 true/false 解析为 bool；0/1 显式兼容
        else:
            raise ValueError(f"字段 {name!r} 应为布尔值，实际 {value!r}（{type(value).__name__}）")
    elif typ is int:
        if isinstance(value, float) and value.is_integer():
            ok_value = int(value)
        elif isinstance(value, int):
            ok_value = value
        else:
            raise ValueError(f"字段 {name!r} 应为整数，实际 {value!r}（{type(value).__name__}）")
    elif typ is float:
        if isinstance(value, (int, float)):
            ok_value = float(value)
        else:
            raise ValueError(f"字段 {name!r} 应为数值，实际 {value!r}（{type(value).__name__}）")
    elif typ is str:
        if isinstance(value, str):
            ok_value = value
        else:
            raise ValueError(f"字段 {name!r} 应为字符串，实际 {value!r}（{type(value).__name__}）")
    else:  # pragma: no cover - 防御性
        ok_value = value

    if bounds is None:
        return ok_value
    if isinstance(bounds, set):
        if ok_value not in bounds:
            raise ValueError(f"字段 {name!r} 取值 {ok_value!r} 不在允许集合 {sorted(bounds)}")
        return ok_value
    lo, hi = bounds
    if not (lo <= ok_value <= hi):
        raise ValueError(f"字段 {name!r} 取值 {ok_value!r} 超出范围 [{lo}, {hi}]")
    return ok_value


def validate_config(config: dict) -> dict:
    """校验 + 兜底 + 未知字段告警。返回补齐默认值后的完整配置（不修改入参）。"""
    if not isinstance(config, dict):
        raise ValueError(f"配置顶层必须是映射(dict)，实际 {type(config).__name__}")

    errors: list[str] = []
    unknowns: list[str] = []
    merged: dict = {}

    for section, defaults in DEFAULTS.items():
        provided = config.get(section, {})
        if not isinstance(provided, dict):
            errors.append(f"段 {section!r} 应为字典，实际 {type(provided).__name__}")
            continue
        merged_section: dict = {}
        for key, default in defaults.items():
            if isinstance(default, dict):  # 子段（env.field / env.reward / algorithm.ablation）
                sub_provided = provided.get(key, {})
                if not isinstance(sub_provided, dict):
                    errors.append(f"段 {section}.{key} 应为字典，实际 {type(sub_provided).__name__}")
                    continue
                sub_merged: dict = {}
                sub_rules = _RULES[f"{section}.{key}"]
                for sub_key, sub_default in default.items():
                    try:
                        sub_merged[sub_key] = _coerce(
                            f"{section}.{key}.{sub_key}", sub_provided.get(sub_key, sub_default), sub_rules[sub_key]
                        )
                    except ValueError as exc:
                        errors.append(str(exc))
                for uk in sorted(set(sub_provided) - set(default)):
                    unknowns.append(f"{section}.{key}.{uk}")
                merged_section[key] = sub_merged
            else:
                try:
                    merged_section[key] = _coerce(f"{section}.{key}", provided.get(key, default), _RULES[section][key])
                except ValueError as exc:
                    errors.append(str(exc))
        for uk in sorted(set(provided) - set(defaults)):
            unknowns.append(f"{section}.{uk}")
        merged[section] = merged_section

    for uk in sorted(set(config) - set(DEFAULTS)):
        unknowns.append(uk)

    if unknowns:
        warnings.warn(f"配置含未知字段（已忽略）：{unknowns}", UserWarning, stacklevel=2)
    if errors:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errors))
    return merged


def load_config(path: str | Path) -> dict:
    """读取 YAML 配置并完成校验与兜底（缺字段取默认、未知字段告警、违例报错）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return validate_config(data)