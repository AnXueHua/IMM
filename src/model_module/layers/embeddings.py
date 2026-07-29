"""时序 embedding 组件集合。

该模块实现多种变量级 token 化方式，包括原始 inverted embedding、patch 化 inverted embedding、混合 embedding 与 decomposition-aware embedding。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .temporal_decomposition import TemporalDecomposition

class PatchEmbedding(nn.Module):
    """把单变量时间序列切成 patch，并投影为 patch token。"""
    def __init__(self, seq_len: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.num_patches = (max(seq_len, patch_len) - patch_len) // stride + 1

        # 每个 patch 的线性投影层
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor):
        # x: [Batch, Time, Vars] -> [Batch, Vars, Time]
        B, T, V = x.shape
        x_inv = x.transpose(1, 2)

        # 分块 (Patching): [Batch, Vars, num_patches, patch_len]
        patches = x_inv.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # 投影 (Projection): [Batch, Vars, num_patches, D_model]
        embeds = self.projection(patches)

        return embeds

class InvertedPatchEmbedding(nn.Module):
    """先对每个变量做 patch 化，再把多个 patch 聚合成单个变量 token。"""
    def __init__(self, seq_len: int, d_model: int, patch_len: int = 16, stride: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(seq_len, patch_len, stride, d_model)

        # 简单的注意力池化，将所有的 patch 聚合成一个 Variable Token
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=-2) # 在 num_patches 维度上进行 Softmax
        )

        # 可选机制: 使用可学习的缺失Token代替简单的全零填充
        self.missing_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.missing_token, std=0.02)

    def forward(self, x: torch.Tensor, sensor_mask: Optional[torch.Tensor] = None):
        # x: [Batch, Time, Vars]
        B, T, V = x.shape

        # 确保时间维度一致
        if T != self.seq_len:
            if T > self.seq_len:
                x = x[:, -self.seq_len:, :]
            else:
                pad_len = self.seq_len - T
                x = nn.functional.pad(x, (0, 0, pad_len, 0))

        # 分块与投影
        # [Batch, Vars, num_patches, D_model]
        patch_embeds = self.patch_embed(x)

        # 对 patch 进行注意力池化
        # [Batch, Vars, num_patches, 1]
        attn_weights = self.attention_pool(patch_embeds)

        # [Batch, Vars, D_model]
        embeds = (patch_embeds * attn_weights).sum(dim=-2)

        if sensor_mask is not None:
            mask_expanded = sensor_mask.unsqueeze(-1).to(embeds.dtype)
            embeds = embeds * mask_expanded + self.missing_token * (1.0 - mask_expanded)
        else:
            sensor_mask = torch.ones(B, V, device=x.device, dtype=torch.bool)

        return embeds, sensor_mask

class InvertedEmbedding(nn.Module):
    """经典倒置 embedding，把整段时间轴直接映射成变量 token。"""
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.projection = nn.Linear(seq_len, d_model)

        self.missing_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.missing_token, std=0.02)

    def forward(self, x: torch.Tensor, sensor_mask: Optional[torch.Tensor] = None):
        B, T, V = x.shape
        if T != self.seq_len:
            if T > self.seq_len:
                x = x[:, -self.seq_len:, :]
            else:
                pad_len = self.seq_len - T
                x = nn.functional.pad(x, (0, 0, pad_len, 0))

        x_inv = x.transpose(1, 2)
        embeds = self.projection(x_inv)

        if sensor_mask is not None:
            mask_expanded = sensor_mask.unsqueeze(-1).to(embeds.dtype)
            embeds = embeds * mask_expanded + self.missing_token * (1.0 - mask_expanded)
        else:
            sensor_mask = torch.ones(B, V, device=x.device, dtype=torch.bool)

        return embeds, sensor_mask


class MixedInvertedEmbedding(nn.Module):
    """融合 raw 路与 patch 路的混合 inverted embedding。"""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        patch_len: int = 16,
        stride: int = 16,
        use_sensor_id_embedding: bool = False,
        max_vars: int = 1024,
        raw_patch_mix_bias: float = -1.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_sensor_id_embedding = use_sensor_id_embedding
        self.max_vars = max_vars

        # Raw path (same core idea as InvertedEmbedding).
        self.raw_projection = nn.Linear(seq_len, d_model)

        # Patch path (same core idea as InvertedPatchEmbedding).
        self.patch_embed = PatchEmbedding(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
        )
        self.patch_attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=-2),
        )

        # Per-variable mix gate: gate * patch + (1-gate) * raw.
        self.mix_gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        with torch.no_grad():
            self.mix_gate[-1].bias.fill_(raw_patch_mix_bias)

        self.sensor_id_embedding = None
        if self.use_sensor_id_embedding:
            self.sensor_id_embedding = nn.Embedding(max_vars, d_model)
            nn.init.normal_(self.sensor_id_embedding.weight, std=0.02)

        self.missing_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.missing_token, std=0.02)

    def _align_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time_steps, n_vars = x.shape
        _ = bsz, n_vars
        if time_steps == self.seq_len:
            return x
        if time_steps > self.seq_len:
            return x[:, -self.seq_len:, :]
        pad_len = self.seq_len - time_steps
        return nn.functional.pad(x, (0, 0, pad_len, 0))

    def _add_sensor_id_embedding(self, embeds: torch.Tensor) -> torch.Tensor:
        if self.sensor_id_embedding is None:
            return embeds
        _, n_vars, _ = embeds.shape
        if n_vars > self.max_vars:
            raise ValueError(
                f"Number of variables ({n_vars}) exceeds max_vars ({self.max_vars}). "
                "Increase max_vars for sensor_id_embedding."
            )
        sensor_ids = torch.arange(n_vars, device=embeds.device)
        sensor_embeds = self.sensor_id_embedding(sensor_ids).unsqueeze(0).to(dtype=embeds.dtype)
        return embeds + sensor_embeds

    def forward(self, x: torch.Tensor, sensor_mask: Optional[torch.Tensor] = None):
        bsz, _, n_vars = x.shape
        x = self._align_seq_len(x)

        # Raw path.
        x_inv = x.transpose(1, 2)  # [B, V, T]
        raw_embeds = self.raw_projection(x_inv)  # [B, V, D]

        # Patch path.
        patch_embeds = self.patch_embed(x)  # [B, V, N, D]
        patch_attn = self.patch_attention_pool(patch_embeds)  # [B, V, N, 1]
        patch_embeds = (patch_embeds * patch_attn).sum(dim=-2)  # [B, V, D]

        # Mix gate.
        gate_input = torch.cat(
            [raw_embeds, patch_embeds, raw_embeds - patch_embeds],
            dim=-1,
        )
        gate = torch.sigmoid(self.mix_gate(gate_input))  # [B, V, 1]
        embeds = gate * patch_embeds + (1.0 - gate) * raw_embeds

        embeds = self._add_sensor_id_embedding(embeds)

        if sensor_mask is not None:
            mask_expanded = sensor_mask.unsqueeze(-1).to(embeds.dtype)
            embeds = embeds * mask_expanded + self.missing_token * (1.0 - mask_expanded)
        else:
            sensor_mask = torch.ones(bsz, n_vars, device=x.device, dtype=torch.bool)

        return embeds, sensor_mask


class DecomposedInvertedEmbedding(nn.Module):
    """先做时序分解，再把 trend/seasonal/residual 三路编码成变量 token。"""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        trend_kernel_size: int = 25,
        seasonal_kernel_size: int = 7,
        seasonal_patch_len: int = 16,
        seasonal_stride: int = 16,
        residual_mode: str = "short_patch",
        residual_patch_len: int = 8,
        residual_stride: int = 8,
        use_sensor_id_embedding: bool = False,
        max_vars: int = 1024,
        gate_hidden_dim: Optional[int] = None,
        return_component_bundle: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_sensor_id_embedding = use_sensor_id_embedding
        self.max_vars = max_vars
        self.return_component_bundle = return_component_bundle

        self.residual_mode = str(residual_mode).lower()
        if self.residual_mode not in {"short_patch", "diff"}:
            raise ValueError("residual_mode must be one of {'short_patch', 'diff'}.")

        seasonal_patch_len = max(1, min(int(seasonal_patch_len), seq_len))
        seasonal_stride = max(1, int(seasonal_stride))
        residual_patch_len = max(1, min(int(residual_patch_len), seq_len))
        residual_stride = max(1, int(residual_stride))

        self.temporal_decomposition = TemporalDecomposition(
            trend_kernel_size=trend_kernel_size,
            seasonal_kernel_size=seasonal_kernel_size,
        )

        # Trend: global linear summary.
        self.trend_projection = nn.Linear(seq_len, d_model)

        # Seasonal: patch summary.
        self.seasonal_patch_embed = PatchEmbedding(
            seq_len=seq_len,
            patch_len=seasonal_patch_len,
            stride=seasonal_stride,
            d_model=d_model,
        )
        self.seasonal_attention_pool = self._build_patch_attention_pool(d_model)

        # Residual: short patch summary or diff summary.
        self.residual_patch_embed = None
        self.residual_attention_pool = None
        self.residual_projection = None
        if self.residual_mode == "short_patch":
            self.residual_patch_embed = PatchEmbedding(
                seq_len=seq_len,
                patch_len=residual_patch_len,
                stride=residual_stride,
                d_model=d_model,
            )
            self.residual_attention_pool = self._build_patch_attention_pool(d_model)
        else:
            self.residual_projection = nn.Linear(seq_len, d_model)

        gate_hidden_dim = gate_hidden_dim if gate_hidden_dim is not None else max(1, d_model // 2)
        self.component_gate = nn.Sequential(
            nn.Linear(3 * d_model, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 3),
        )
        with torch.no_grad():
            self.component_gate[-1].bias.copy_(torch.tensor([0.5, 0.5, -1.0]))

        self.sensor_id_embedding = None
        if self.use_sensor_id_embedding:
            self.sensor_id_embedding = nn.Embedding(max_vars, d_model)
            nn.init.normal_(self.sensor_id_embedding.weight, std=0.02)

        self.missing_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.missing_token, std=0.02)

    @staticmethod
    def _build_patch_attention_pool(d_model: int) -> nn.Sequential:
        hidden_dim = max(1, d_model // 2)
        return nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=-2),
        )

    def _align_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time_steps, n_vars = x.shape
        _ = bsz, n_vars
        if time_steps == self.seq_len:
            return x
        if time_steps > self.seq_len:
            return x[:, -self.seq_len:, :]
        pad_len = self.seq_len - time_steps
        return nn.functional.pad(x, (0, 0, pad_len, 0))

    def _patch_summary(
        self,
        signal: torch.Tensor,
        patch_embed: PatchEmbedding,
        attention_pool: nn.Sequential,
    ) -> torch.Tensor:
        patch_tokens = patch_embed(signal)
        patch_attn = attention_pool(patch_tokens)
        return (patch_tokens * patch_attn).sum(dim=-2)

    def _encode_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if self.residual_mode == "short_patch":
            if self.residual_patch_embed is None or self.residual_attention_pool is None:
                raise RuntimeError("Residual short-patch modules are not initialized.")
            return self._patch_summary(
                signal=residual,
                patch_embed=self.residual_patch_embed,
                attention_pool=self.residual_attention_pool,
            )

        diff = residual[:, 1:, :] - residual[:, :-1, :]
        diff = nn.functional.pad(diff, (0, 0, 1, 0))
        diff_inv = diff.transpose(1, 2)
        if self.residual_projection is None:
            raise RuntimeError("Residual diff projection is not initialized.")
        return self.residual_projection(diff_inv)

    def _add_sensor_id_embedding(self, embeds: torch.Tensor) -> torch.Tensor:
        if self.sensor_id_embedding is None:
            return embeds
        _, n_vars, _ = embeds.shape
        if n_vars > self.max_vars:
            raise ValueError(
                f"Number of variables ({n_vars}) exceeds max_vars ({self.max_vars}). "
                "Increase max_vars for sensor_id_embedding."
            )
        sensor_ids = torch.arange(n_vars, device=embeds.device)
        sensor_embeds = self.sensor_id_embedding(sensor_ids).unsqueeze(0).to(dtype=embeds.dtype)
        return embeds + sensor_embeds

    def forward(
        self,
        x: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
        return_component_bundle: Optional[bool] = None,
    ):
        bsz, _, n_vars = x.shape
        x = self._align_seq_len(x)

        trend, seasonal, residual = self.temporal_decomposition(x)

        trend_embeds = self.trend_projection(trend.transpose(1, 2))
        seasonal_embeds = self._patch_summary(
            signal=seasonal,
            patch_embed=self.seasonal_patch_embed,
            attention_pool=self.seasonal_attention_pool,
        )
        residual_embeds = self._encode_residual(residual)

        gate_input = torch.cat([trend_embeds, seasonal_embeds, residual_embeds], dim=-1)
        component_gate = torch.softmax(self.component_gate(gate_input), dim=-1)

        embeds = (
            component_gate[..., 0:1] * trend_embeds
            + component_gate[..., 1:2] * seasonal_embeds
            + component_gate[..., 2:3] * residual_embeds
        )
        embeds = self._add_sensor_id_embedding(embeds)

        if sensor_mask is not None:
            mask_expanded = sensor_mask.unsqueeze(-1).to(embeds.dtype)
            embeds = embeds * mask_expanded + self.missing_token * (1.0 - mask_expanded)
        else:
            sensor_mask = torch.ones(bsz, n_vars, device=x.device, dtype=torch.bool)

        should_return_bundle = (
            self.return_component_bundle if return_component_bundle is None else return_component_bundle
        )
        if not should_return_bundle:
            return embeds, sensor_mask

        component_bundle: Dict[str, torch.Tensor] = {
            "trend_series": trend,
            "seasonal_series": seasonal,
            "residual_series": residual,
            "trend_embed": trend_embeds,
            "seasonal_embed": seasonal_embeds,
            "residual_embed": residual_embeds,
            "component_gate": component_gate,
        }
        return embeds, sensor_mask, component_bundle
