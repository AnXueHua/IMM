"""非倒置时间细节分支。

该模块不再只输出 pooled `[B, V, D]` 特征，而是显式保留层级时间状态：

1. `local_tokens`
2. `segment_summaries`
3. `global_temporal_memory`
4. `summary`

默认仍返回 `summary`，以兼容旧调用；当 `return_token_bundle=True` 时，
返回完整层级字典，供 `time_stream` 主预测、retrieval 和后续语义桥接直接消费。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDetailRoute(nn.Module):
    """对每个传感器独立进行层级时间编码。"""

    VALID_STATE_FUSION_MODES = {
        "all",
        "local_only",
        "segment_only",
        "global_only",
        "no_segment",
        "no_global",
    }

    def __init__(
        self,
        d_model: int,
        patch_len: int = 16,
        stride: int = 8,
        hidden_channels: int = 32,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_layers: int = 1,
        ff_mult: int = 2,
        max_tokens: int = 128,
        max_position_embeddings: int = 256,
        segment_size: int = 4,
        segment_kernel_size: int = 3,
        global_memory_slots: int = 4,
        state_fusion_mode: str = "all",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.max_tokens = max_tokens
        self.max_position_embeddings = max_position_embeddings
        self.segment_size = max(1, int(segment_size))
        self.global_memory_slots = max(1, int(global_memory_slots))
        self.hidden_channels = max(1, int(hidden_channels))
        self.state_fusion_mode = str(state_fusion_mode).lower()
        if self.state_fusion_mode not in self.VALID_STATE_FUSION_MODES:
            raise ValueError(
                f"Unsupported temporal state fusion mode={state_fusion_mode}. "
                f"Expected one of {sorted(self.VALID_STATE_FUSION_MODES)}"
            )

        # local token 构造。
        self.patch_norm = nn.LayerNorm(patch_len)
        self.patch_hidden = nn.Linear(patch_len, self.hidden_channels, bias=False)
        self.patch_activation = nn.GELU()
        self.patch_proj = nn.Linear(self.hidden_channels, d_model, bias=False)
        self.input_dropout = nn.Dropout(dropout)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, max_position_embeddings, d_model))

        nhead = self._resolve_nhead(d_model=d_model, requested=num_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # segment_summaries: temporal conv + gated pooling。
        conv_padding = max(0, int(segment_kernel_size) // 2)
        self.segment_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=max(1, int(segment_kernel_size)),
            padding=conv_padding,
            groups=d_model,
            bias=False,
        )
        self.segment_gate = nn.Linear(d_model, 1, bias=False)
        self.segment_proj = nn.Linear(d_model, d_model, bias=False)
        self.segment_norm = nn.LayerNorm(d_model)

        # global_temporal_memory: learned memory slots。
        self.global_queries = nn.Parameter(torch.zeros(1, self.global_memory_slots, d_model))
        self.global_proj = nn.Linear(d_model, d_model, bias=False)
        self.global_norm = nn.LayerNorm(d_model)

        # summary 聚合。
        self.local_attn_pool = nn.Linear(d_model, 1, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        # 缺失传感器占位。
        self.missing_token = nn.Parameter(torch.zeros(1, 1, d_model))

        nn.init.trunc_normal_(self.patch_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.global_queries, std=0.02)
        nn.init.normal_(self.missing_token, std=0.02)

    @staticmethod
    def _resolve_nhead(d_model: int, requested: int) -> int:
        """自动修正可被 `d_model` 整除的注意力头数。"""

        if requested <= 1:
            return 1
        nhead = min(requested, d_model)
        while nhead > 1 and d_model % nhead != 0:
            nhead -= 1
        return max(1, nhead)

    def _extract_patches(self, x_var: torch.Tensor) -> torch.Tensor:
        """把单变量序列切成 patch，并在长序列时做自适应压缩。"""

        if x_var.size(-1) < self.patch_len:
            pad = self.patch_len - x_var.size(-1)
            x_var = F.pad(x_var.unsqueeze(1), (0, pad), mode="replicate").squeeze(1)

        patches = x_var.unsqueeze(1).unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.squeeze(1).contiguous()

        token_count = patches.size(1)
        if token_count > self.max_tokens:
            patches_t = patches.transpose(1, 2)
            pooled_mean = F.adaptive_avg_pool1d(patches_t, self.max_tokens)
            pooled_max = F.adaptive_max_pool1d(patches_t, self.max_tokens)
            patches = (0.5 * (pooled_mean + pooled_max)).transpose(1, 2).contiguous()
        return patches

    def _position_embedding(self, n_tokens: int, dtype: torch.dtype) -> torch.Tensor:
        """根据 token 数裁剪或插值位置编码。"""

        if n_tokens <= self.max_position_embeddings:
            return self.patch_pos_embed[:, :n_tokens, :].to(dtype=dtype)
        pos = self.patch_pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=n_tokens, mode="linear", align_corners=False)
        return pos.transpose(1, 2).to(dtype=dtype)

    def _build_local_tokens(self, x_var: torch.Tensor) -> torch.Tensor:
        """构建 `[B*V, P, D]` 的局部时间 token。"""

        patches = self._extract_patches(x_var)
        normalized_patches = self.patch_norm(patches)
        tokens = self.patch_hidden(normalized_patches)
        tokens = self.patch_activation(tokens)
        tokens = self.patch_proj(tokens)
        tokens = tokens + self._position_embedding(tokens.size(1), dtype=tokens.dtype)
        tokens = self.input_dropout(tokens)
        return self.temporal_encoder(tokens)

    def _build_segment_summaries(self, local_tokens: torch.Tensor) -> torch.Tensor:
        """用 `temporal conv + gated pooling` 生成段级状态。"""

        conv_tokens = self.segment_conv(local_tokens.transpose(1, 2)).transpose(1, 2)
        merged_tokens = local_tokens + conv_tokens

        token_count = merged_tokens.size(1)
        if token_count == 0:
            return merged_tokens

        pad_needed = (self.segment_size - (token_count % self.segment_size)) % self.segment_size
        if pad_needed > 0:
            pad_tensor = merged_tokens[:, -1:, :].expand(-1, pad_needed, -1)
            merged_tokens = torch.cat([merged_tokens, pad_tensor], dim=1)

        num_segments = merged_tokens.size(1) // self.segment_size
        grouped = merged_tokens.reshape(
            merged_tokens.size(0),
            num_segments,
            self.segment_size,
            self.d_model,
        )
        gate_scores = self.segment_gate(grouped).squeeze(-1)
        gate_weights = torch.softmax(gate_scores, dim=-1)
        segment = torch.sum(grouped * gate_weights.unsqueeze(-1), dim=2)
        segment = self.segment_norm(self.segment_proj(segment))
        return segment

    def _build_global_temporal_memory(self, segment_summaries: torch.Tensor) -> torch.Tensor:
        """用 learned memory slots 聚合长期时间上下文。"""

        num_items = segment_summaries.size(0)
        if segment_summaries.size(1) == 0:
            base_memory = self.global_queries.expand(num_items, -1, -1)
            return self.global_norm(self.global_proj(base_memory))

        queries = self.global_queries.expand(num_items, -1, -1)
        logits = torch.matmul(queries, segment_summaries.transpose(1, 2))
        logits = logits / math.sqrt(float(self.d_model))
        weights = torch.softmax(logits, dim=-1)
        memory = torch.matmul(weights, segment_summaries)
        return self.global_norm(self.global_proj(memory))

    def _pool_local_tokens(self, local_tokens: torch.Tensor) -> torch.Tensor:
        """从局部 token 提取局部摘要。"""

        scores = self.local_attn_pool(local_tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        attn_pool = torch.sum(local_tokens * weights.unsqueeze(-1), dim=1)
        mean_pool = local_tokens.mean(dim=1)
        max_pool = local_tokens.amax(dim=1)
        return (attn_pool + mean_pool + max_pool) / 3.0

    def _build_summary(
        self,
        local_tokens: torch.Tensor,
        segment_summaries: torch.Tensor,
        global_temporal_memory: torch.Tensor,
    ) -> torch.Tensor:
        """融合 local / segment / global 三层信息得到 `[B*V, D]`。"""

        local_summary = self._pool_local_tokens(local_tokens)
        segment_summary = segment_summaries.mean(dim=1) if segment_summaries.size(1) > 0 else None
        global_summary = global_temporal_memory.mean(dim=1) if global_temporal_memory.size(1) > 0 else None

        mode = self.state_fusion_mode
        if mode == "local_only":
            fused = local_summary
        elif mode == "segment_only":
            if segment_summary is None:
                raise RuntimeError("segment_only fusion requires segment summaries")
            fused = segment_summary
        elif mode == "global_only":
            if global_summary is None:
                raise RuntimeError("global_only fusion requires global temporal memory")
            fused = global_summary
        else:
            components = [local_summary]
            if mode != "no_segment" and segment_summary is not None:
                components.append(segment_summary)
            if mode != "no_global" and global_summary is not None:
                components.append(global_summary)
            fused = torch.stack(components, dim=0).mean(dim=0)
        return self.norm(self.output_proj(fused))

    def _empty_state_tokens(self, reference: torch.Tensor) -> torch.Tensor:
        """返回与当前 batch/device/dtype 对齐的空层级状态。"""

        return reference.new_zeros(reference.size(0), 0, self.d_model)

    def _mask_features(self, features: torch.Tensor, sensor_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """把缺失传感器替换成可学习 missing token。"""

        if sensor_mask is None:
            return features
        mask_expanded = sensor_mask.unsqueeze(-1).to(features.dtype)
        return features * mask_expanded + self.missing_token * (1.0 - mask_expanded)

    def _mask_time_tensor(self, tensor: torch.Tensor, sensor_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """对 `[B, V, ...]` 形状的时间状态施加传感器掩码。"""

        if sensor_mask is None:
            return tensor
        mask = sensor_mask
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(-1)
        return tensor * mask.to(tensor.dtype)

    def _build_time_state(
        self,
        x: torch.Tensor,
        sensor_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """构建完整层级 `time_state`。"""

        batch_size, _, num_vars = x.shape
        x_var = x.transpose(1, 2).reshape(batch_size * num_vars, -1)

        local_tokens = self._build_local_tokens(x_var)

        mode = self.state_fusion_mode
        if mode in {"local_only", "no_segment"}:
            segment_summaries = self._empty_state_tokens(local_tokens)
        else:
            segment_summaries = self._build_segment_summaries(local_tokens)

        if mode in {"local_only", "segment_only", "no_global"}:
            global_temporal_memory = self._empty_state_tokens(local_tokens)
        elif mode == "no_segment":
            global_temporal_memory = self._build_global_temporal_memory(local_tokens)
        else:
            global_temporal_memory = self._build_global_temporal_memory(segment_summaries)
        summary = self._build_summary(local_tokens, segment_summaries, global_temporal_memory)

        local_count = local_tokens.size(1)
        segment_count = segment_summaries.size(1)
        global_count = global_temporal_memory.size(1)

        local_tokens = local_tokens.reshape(batch_size, num_vars, local_count, self.d_model)
        segment_summaries = segment_summaries.reshape(batch_size, num_vars, segment_count, self.d_model)
        global_temporal_memory = global_temporal_memory.reshape(batch_size, num_vars, global_count, self.d_model)
        summary = summary.reshape(batch_size, num_vars, self.d_model)

        summary = self._mask_features(summary, sensor_mask)
        local_tokens = self._mask_time_tensor(local_tokens, sensor_mask)
        segment_summaries = self._mask_time_tensor(segment_summaries, sensor_mask)
        global_temporal_memory = self._mask_time_tensor(global_temporal_memory, sensor_mask)

        return {
            "local_tokens": local_tokens,
            "segment_summaries": segment_summaries,
            "global_temporal_memory": global_temporal_memory,
            "summary": summary,
        }

    def forward(
        self,
        x: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
        return_token_bundle: bool = False,
    ) -> Any:
        """默认返回兼容旧接口的 `summary`，可选返回完整层级状态。"""

        time_state = self._build_time_state(x, sensor_mask)
        summary = time_state["summary"]

        if not return_token_bundle:
            return summary

        local_tokens = time_state["local_tokens"]
        return {
            "features": summary,
            "tokens": local_tokens,
            "num_tokens": local_tokens.size(2),
            "local_tokens": local_tokens,
            "segment_summaries": time_state["segment_summaries"],
            "global_temporal_memory": time_state["global_temporal_memory"],
            "summary": summary,
            "time_state": time_state,
        }
