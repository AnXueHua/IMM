"""时间序列因果分解层。

该模块把输入序列拆成趋势项、季节项和残差项，
并确保每一步平滑只依赖当前及历史时刻，避免未来信息泄漏。
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDecomposition(nn.Module):
    """对 `[B, T, V]` 输入执行因果友好的加性分解。"""

    def __init__(
        self,
        trend_kernel_size: int = 25,
        seasonal_kernel_size: int = 7,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        self.trend_kernel_size = int(trend_kernel_size)
        self.seasonal_kernel_size = int(seasonal_kernel_size)
        self.padding_mode = padding_mode

        if self.trend_kernel_size <= 0:
            raise ValueError("trend_kernel_size must be a positive integer.")
        if self.seasonal_kernel_size <= 0:
            raise ValueError("seasonal_kernel_size must be a positive integer.")
        if self.padding_mode not in {"replicate", "zeros"}:
            raise ValueError("padding_mode must be one of {'replicate', 'zeros'}.")

    def _causal_moving_average(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """计算只依赖历史窗口的移动平均。"""

        if kernel_size <= 1:
            return x

        # 通过前缀和实现滑动平均，避免显式循环并确保无未来泄漏。
        x_bvt = x.transpose(1, 2)
        cumsum = torch.cumsum(x_bvt, dim=-1)
        padded_cumsum = F.pad(cumsum, (1, 0), mode="constant", value=0.0)

        time_steps = x_bvt.size(-1)
        end_idx = torch.arange(1, time_steps + 1, device=x.device)
        start_idx = torch.clamp(end_idx - kernel_size, min=0)

        window_sum = padded_cumsum[:, :, end_idx] - padded_cumsum[:, :, start_idx]
        window_len = (end_idx - start_idx).to(x.dtype).view(1, 1, -1)
        smoothed = window_sum / window_len
        return smoothed.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回趋势、季节与残差三部分。"""

        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, T, V], but got ndim={x.ndim}.")

        trend = self._causal_moving_average(x, self.trend_kernel_size)
        detrended = x - trend
        seasonal = self._causal_moving_average(detrended, self.seasonal_kernel_size)
        residual = x - trend - seasonal
        return trend, seasonal, residual
