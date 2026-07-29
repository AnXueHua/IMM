"""Forecasting 预测头。

本模块保留当前 v2 主线需要的数值分支：

1. `TimeForecastHead` 从层级 `time_state` 生成 `y_time_base`。
2. `SensorForecastHead` 从 `sensor_state` 生成 `y_sensor_base`。
3. `RelationCorrectionHead` 从 `sensor_state` 生成 `delta_relation`。
4. `EvidenceCorrectionHead` 根据 `retrieval_evidence` 生成 `delta_evidence`。

LLM 不在这里直接参与数值修正；后续动态控制只能通过独立 router 使用 sidecar 状态。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def _pool_time_level(level_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """把 `[B, V, T, D]` 池化为 `[B, V, D]`，并兼容已池化输入。"""

    if level_tensor is None or not torch.is_tensor(level_tensor):
        return None
    if level_tensor.dim() == 4:
        if level_tensor.size(2) == 0:
            return None
        return level_tensor.mean(dim=2)
    if level_tensor.dim() == 3:
        return level_tensor
    return None


class SensorForecastHead(nn.Module):
    """从 sensor-centric 表征生成基础预测。"""

    def __init__(self, d_model: int, pred_len: int, dropout: float = 0.1, head_mode: str = "plain_linear") -> None:
        super().__init__()
        self.head_mode = head_mode
        if head_mode == "plain_linear":
            self.proj = nn.Linear(d_model, pred_len)
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, pred_len),
            )

    def forward(self, sensor_state: torch.Tensor) -> torch.Tensor:
        """将 `[B, V, D]` 的传感器状态投影为 `[B, pred_len, V]`。"""

        return self.proj(sensor_state).transpose(1, 2)


class TimeForecastHead(nn.Module):
    """从层级时间状态生成基础预测。"""

    def __init__(self, d_model: int, pred_len: int, dropout: float = 0.1, fusion_mode: str = "learned") -> None:
        super().__init__()
        self.pred_len = pred_len
        if fusion_mode != "learned":
            raise ValueError("TimeForecastHead now only supports learned time-level fusion")

        self.local_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        self.segment_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, pred_len),
        )
        # 多层级时间状态权重由训练学习，旧固定 preset 已归档为失败尝试。
        self.time_level_logits = nn.Parameter(torch.tensor([1.0, -0.25, -0.50, 0.75], dtype=torch.float32))

    def _project_level(
        self,
        pooled_level: Optional[torch.Tensor],
        projector: nn.Module,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        """把单层时间特征投影为 `[B, pred_len, V]`。"""

        features = pooled_level if pooled_level is not None else fallback
        return projector(features).transpose(1, 2)

    def forward(
        self,
        time_state: Dict[str, torch.Tensor],
        fallback_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """融合 local / segment / global / summary 四类时间预测。"""

        summary = time_state.get("summary")
        if summary is None or not torch.is_tensor(summary):
            summary = fallback_features

        local_tokens = _pool_time_level(time_state.get("local_tokens"))
        segment_summaries = _pool_time_level(time_state.get("segment_summaries"))
        global_temporal_memory = _pool_time_level(time_state.get("global_temporal_memory"))

        y_local = self._project_level(local_tokens, self.local_proj, summary)
        y_segment = self._project_level(segment_summaries, self.segment_proj, summary)
        y_global = self._project_level(global_temporal_memory, self.global_proj, summary)
        y_summary = self.summary_proj(summary).transpose(1, 2)

        fusion_weights = torch.softmax(self.time_level_logits, dim=0)

        y_fused = (
            fusion_weights[0] * y_local
            + fusion_weights[1] * y_segment
            + fusion_weights[2] * y_global
            + fusion_weights[3] * y_summary
        )
        return {
            "y_time_base": y_fused,
            "y_local": y_local,
            "y_segment": y_segment,
            "y_global": y_global,
            "y_summary": y_summary,
            "time_level_weights": fusion_weights.detach(),
        }


class RelationCorrectionHead(nn.Module):
    """从跨传感器状态生成关系修正量。"""

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        relation_code_dim: int = 32,
        use_time_summary: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = d_model * 2 if use_time_summary else d_model
        self.use_time_summary = use_time_summary
        self.relation_encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, relation_code_dim),
        )
        self.horizon_projector = nn.Linear(relation_code_dim, pred_len)
        self.gate_projector = nn.Linear(relation_code_dim, pred_len)

    def forward(
        self,
        sensor_state: torch.Tensor,
        time_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """输出 `[B, pred_len, V]` 的关系修正量。"""

        if self.use_time_summary and time_summary is not None:
            features = torch.cat([sensor_state, time_summary], dim=-1)
        else:
            features = sensor_state
        relation_code = self.relation_encoder(features)
        raw_delta = self.horizon_projector(relation_code).transpose(1, 2)
        gate = torch.sigmoid(self.gate_projector(relation_code)).transpose(1, 2)
        return torch.tanh(raw_delta) * gate


class EvidenceCorrectionHead(nn.Module):
    """根据 retrieval evidence 产生证据修正量。"""

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        evidence_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.retrieval_stats_proj = nn.Linear(12, d_model)
        self.evidence_mixer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, evidence_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(evidence_hidden_dim, d_model),
        )
        self.out_proj = nn.Linear(d_model, pred_len)
        self.gate_proj = nn.Linear(d_model, pred_len)

    def _extract_retrieval_context(
        self,
        retrieval_evidence: Optional[dict],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """从 retrieval bundle 中提取 `[B, V, D]` 证据上下文。"""

        if not isinstance(retrieval_evidence, dict):
            return torch.zeros_like(reference)

        evidence_summary = retrieval_evidence.get("retrieval_evidence_summary")
        if torch.is_tensor(evidence_summary):
            evidence_summary = evidence_summary.to(device=reference.device, dtype=reference.dtype)
            if evidence_summary.dim() == 2 and evidence_summary.size(-1) == reference.size(-1):
                return evidence_summary.unsqueeze(1).expand(-1, reference.size(1), -1)

        evidence_tokens = retrieval_evidence.get("retrieval_evidence_tokens")
        if torch.is_tensor(evidence_tokens):
            evidence_tokens = evidence_tokens.to(device=reference.device, dtype=reference.dtype)
            if evidence_tokens.dim() == 3 and evidence_tokens.size(-1) == reference.size(-1):
                pooled_tokens = evidence_tokens.mean(dim=1, keepdim=True)
                return pooled_tokens.expand(-1, reference.size(1), -1)

        retrieval_context = retrieval_evidence.get("retrieval_context")
        if torch.is_tensor(retrieval_context):
            retrieval_context = retrieval_context.to(device=reference.device, dtype=reference.dtype)
            if retrieval_context.dim() == 2:
                retrieval_context = retrieval_context.unsqueeze(1).expand(-1, reference.size(1), -1)
            elif retrieval_context.dim() == 3 and retrieval_context.size(1) != reference.size(1):
                retrieval_context = retrieval_context.mean(dim=1, keepdim=True).expand(-1, reference.size(1), -1)
            if retrieval_context.dim() == 3 and retrieval_context.size(-1) == reference.size(-1):
                return retrieval_context

        retrieval_stats = retrieval_evidence.get("retrieval_stats")
        if torch.is_tensor(retrieval_stats):
            retrieval_stats = retrieval_stats.to(device=reference.device, dtype=reference.dtype)
            stats_dim = retrieval_stats.size(-1)
            if stats_dim < 12:
                pad = torch.zeros(
                    retrieval_stats.size(0),
                    12 - stats_dim,
                    device=reference.device,
                    dtype=reference.dtype,
                )
                retrieval_stats = torch.cat([retrieval_stats, pad], dim=-1)
            elif stats_dim > 12:
                retrieval_stats = retrieval_stats[:, :12]
            stats_context = self.retrieval_stats_proj(retrieval_stats).unsqueeze(1)
            return stats_context.expand(-1, reference.size(1), -1)

        return torch.zeros_like(reference)

    def forward(
        self,
        retrieval_evidence: Optional[dict],
        reference_features: torch.Tensor,
    ) -> torch.Tensor:
        """输出 `[B, pred_len, V]` 的证据修正量。"""

        retrieval_context = self._extract_retrieval_context(retrieval_evidence, reference_features)
        mixed = self.evidence_mixer(retrieval_context)
        raw_delta = self.out_proj(mixed).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(mixed)).transpose(1, 2)
        return torch.tanh(raw_delta) * gate
