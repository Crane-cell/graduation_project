# -*- coding: utf-8 -*-
"""训练曲线记录（F18）：TensorBoard 标量封装。"""
from torch.utils.tensorboard import SummaryWriter


class MetricLogger:
    """向 TensorBoard 写标量；Day06/Day08 的报告数据不取自此处的显示值，
    而是来自工具真实输出（pytest-cov / 计时脚本）。
    """

    def __init__(self, log_dir: str) -> None:
        self._writer = SummaryWriter(log_dir)

    def write(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, float(value), step)

    def close(self) -> None:
        self._writer.close()
