# -*- coding: utf-8 -*-
"""全局随机种子（NF-02 可复现性）：Python / NumPy / PyTorch 三方统一固定。

用法：训练入口处调用一次 set_global_seed(cfg["train"]["seed"])。
"""
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """固定全部随机源（F09：固定种子下两次运行轨迹完全一致）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
