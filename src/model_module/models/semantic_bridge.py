"""语义桥接模块。

该模块把时序主状态、传感器关系状态和 retrieval 证据整理成统一的 bridge tokens，
为后续 LLM 轻量适配和 evidence correction 提供共享语义接口。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def _limit_token_count(tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """限制 token 数量，避免桥接序列过长。"""

    if tokens.size(1) <= max_tokens:
        return tokens
    return tokens[:, :max_tokens, :]


class SemanticBridge(nn.Module):
    """把多路状态压成统一的语义 bridge。"""

    def __init__(
        self,
        d_model: int,
        stats_dim: int,
        max_time_tokens: int = 8,
        max_evidence_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_time_tokens = max(1, int(max_time_tokens))
        self.max_evidence_tokens = max(1, int(max_evidence_tokens))

        self.time_token_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.relation_token_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.evidence_token_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stats_token_proj = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 用 token type embedding 区分时间、关系、证据和统计四类语义来源。
        self.token_type_embed = nn.Embedding(4, d_model)
        self.bridge_pool = nn.Linear(d_model, 1, bias=False)
        self.bridge_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _fallback_summary(summary: Optional[torch.Tensor], sensor_state: torch.Tensor) -> torch.Tensor:
        """保证 time summary 始终存在。"""

        if torch.is_tensor(summary):
            return summary
        return sensor_state

    def _build_time_event_tokens(
        self,
        time_state: Dict[str, Optional[torch.Tensor]],
        sensor_state: torch.Tensor,
    ) -> torch.Tensor:
        """从 segment/global 时间状态中构造事件级 token。"""

        segment_summaries = time_state.get("segment_summaries")
        if torch.is_tensor(segment_summaries) and segment_summaries.dim() == 4:
            # 先按传感器求均值，保留段级时间 token。
            time_tokens = segment_summaries.mean(dim=1)
        else:
            summary = self._fallback_summary(time_state.get("summary"), sensor_state)
            time_tokens = summary.mean(dim=1, keepdim=True)

        global_memory = time_state.get("global_temporal_memory")
        if torch.is_tensor(global_memory) and global_memory.dim() == 4:
            global_tokens = global_memory.mean(dim=1)
            time_tokens = torch.cat([time_tokens, global_tokens], dim=1)

        time_tokens = _limit_token_count(time_tokens, self.max_time_tokens)
        return self.time_token_proj(time_tokens)

    def _build_relation_tokens(self, sensor_state: torch.Tensor) -> torch.Tensor:
        """从 sensor_state 提取关系语义 token。"""

        mean_token = sensor_state.mean(dim=1, keepdim=True)
        max_token = sensor_state.amax(dim=1, keepdim=True)
        std_token = sensor_state.std(dim=1, unbiased=False, keepdim=True)
        relation_tokens = torch.cat([mean_token, max_token, std_token], dim=1)
        return self.relation_token_proj(relation_tokens)

    def _build_evidence_tokens(
        self,
        retrieval_evidence: Optional[dict],
        sensor_state: torch.Tensor,
    ) -> torch.Tensor:
        """从 retrieval bundle 提取证据 token。"""

        if not isinstance(retrieval_evidence, dict):
            return sensor_state.new_zeros(sensor_state.size(0), 0, sensor_state.size(-1))

        token_parts = []
        candidate_tokens = retrieval_evidence.get("retrieval_evidence_tokens")
        if torch.is_tensor(candidate_tokens) and candidate_tokens.dim() == 3:
            candidate_tokens = candidate_tokens.to(device=sensor_state.device, dtype=sensor_state.dtype)
        else:
            candidate_tokens = sensor_state.new_zeros(sensor_state.size(0), 0, sensor_state.size(-1))

        reserved_tokens = 0
        evidence_summary = retrieval_evidence.get("retrieval_evidence_summary")
        if torch.is_tensor(evidence_summary):
            reserved_tokens += 1

        retrieval_context = retrieval_evidence.get("retrieval_context")
        if torch.is_tensor(retrieval_context) and retrieval_context.dim() == 3:
            reserved_tokens += 1

        candidate_budget = max(0, self.max_evidence_tokens - reserved_tokens)
        if candidate_tokens.size(1) > 0 and candidate_budget > 0:
            token_parts.append(_limit_token_count(candidate_tokens, candidate_budget))

        if torch.is_tensor(evidence_summary):
            summary_token = evidence_summary.to(device=sensor_state.device, dtype=sensor_state.dtype).unsqueeze(1)
            token_parts.append(summary_token)

        if torch.is_tensor(retrieval_context) and retrieval_context.dim() == 3:
            pooled_context = retrieval_context.mean(dim=1, keepdim=True).to(
                device=sensor_state.device,
                dtype=sensor_state.dtype,
            )
            token_parts.append(pooled_context)

        if token_parts:
            evidence_tokens = torch.cat(token_parts, dim=1)
        else:
            evidence_tokens = sensor_state.new_zeros(sensor_state.size(0), 0, sensor_state.size(-1))

        evidence_tokens = _limit_token_count(evidence_tokens, self.max_evidence_tokens)
        if evidence_tokens.size(1) == 0:
            return evidence_tokens
        return self.evidence_token_proj(evidence_tokens)

    def _build_stats_token(self, semantic_stats: torch.Tensor) -> torch.Tensor:
        """把统计语义向量变成一个独立 token。"""

        return self.stats_token_proj(semantic_stats).unsqueeze(1)

    def _inject_token_type(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        """为不同来源 token 注入 type embedding。"""

        if tokens.size(1) == 0:
            return tokens
        type_embed = self.token_type_embed.weight[type_id].view(1, 1, -1).to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens + type_embed

    def _pool_bridge_tokens(self, bridge_tokens: torch.Tensor) -> torch.Tensor:
        """对 bridge tokens 做注意力池化，得到样本级桥接摘要。"""

        scores = self.bridge_pool(bridge_tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(bridge_tokens * weights.unsqueeze(-1), dim=1)
        return self.bridge_norm(pooled)

    def forward(
        self,
        time_state: Dict[str, Optional[torch.Tensor]],
        sensor_state: torch.Tensor,
        retrieval_evidence: Optional[dict],
        semantic_stats: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """输出桥接 token 及其摘要。"""

        time_tokens = self._inject_token_type(
            self._build_time_event_tokens(time_state=time_state, sensor_state=sensor_state),
            type_id=0,
        )
        relation_tokens = self._inject_token_type(
            self._build_relation_tokens(sensor_state),
            type_id=1,
        )
        evidence_tokens = self._inject_token_type(
            self._build_evidence_tokens(retrieval_evidence, sensor_state),
            type_id=2,
        )
        stats_token = self._inject_token_type(
            self._build_stats_token(semantic_stats),
            type_id=3,
        )

        token_groups = [time_tokens, relation_tokens, stats_token]
        if evidence_tokens.size(1) > 0:
            token_groups.append(evidence_tokens)
        bridge_tokens = torch.cat(token_groups, dim=1)
        bridge_summary = self._pool_bridge_tokens(bridge_tokens)

        return {
            "time_event_tokens": time_tokens,
            "relation_tokens": relation_tokens,
            "evidence_tokens": evidence_tokens,
            "stats_token": stats_token,
            "bridge_tokens": bridge_tokens,
            "bridge_summary": bridge_summary,
        }

