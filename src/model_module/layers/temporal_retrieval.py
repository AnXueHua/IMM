"""时间证据检索模块。

该模块把 retrieval 固定为显式证据流，而不是 future prior 生成器：

1. 检索相似历史片段
2. 产出 `retrieval_evidence_tokens`
3. 产出 `retrieval_evidence_summary`
4. 仅输出预测链和 sidecar 所需的证据字段，不再保留 future-prior 风格的序列上下文
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPatternRetriever(nn.Module):
    """基于历史原型的 retrieval evidence 构造器。"""

    RETRIEVAL_STATS_DIM = 12

    def __init__(
        self,
        d_model: int,
        num_slots: int = 128,
        top_k: int = 4,
        key_dim: int = 64,
        key_downsample_len: int = 128,
        max_vars: int = 512,
        confidence_temperature: float = 1.0,
        confidence_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_slots = max(1, int(num_slots))
        self.top_k = max(1, int(top_k))
        self.key_dim = max(1, int(key_dim))
        self.key_downsample_len = max(1, int(key_downsample_len))
        self.max_vars = max(1, int(max_vars))
        self.confidence_temperature = max(float(confidence_temperature), 1e-4)
        self.confidence_bias = float(confidence_bias)

        self.series_key_proj = nn.Sequential(
            nn.Linear(self.key_downsample_len * 3, self.key_dim),
            nn.GELU(),
            nn.Linear(self.key_dim, self.key_dim),
        )
        self.feature_key_proj = nn.Sequential(
            nn.Linear(self.d_model * 3, self.key_dim),
            nn.GELU(),
            nn.Linear(self.key_dim, self.key_dim),
        )
        self.sensor_value_proj = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.stats_value_proj = nn.Sequential(
            nn.Linear(self.RETRIEVAL_STATS_DIM, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.numeric_value_proj = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.evidence_norm = nn.LayerNorm(self.d_model)

        self.register_buffer("prototype_keys", torch.zeros(self.num_slots, self.key_dim))
        self.register_buffer("prototype_values", torch.zeros(self.num_slots, self.max_vars, self.d_model))
        self.register_buffer("prototype_stats", torch.zeros(self.num_slots, self.RETRIEVAL_STATS_DIM))
        self.register_buffer("prototype_valid_mask", torch.zeros(self.num_slots, dtype=torch.bool))
        self.register_buffer("prototype_write_ptr", torch.zeros(1, dtype=torch.long))

    def _downsample_series(self, series: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool1d(series.unsqueeze(1), self.key_downsample_len).squeeze(1)

    def _lag1_correlation(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) <= 1:
            return torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        x_prev = x[:, :-1, :]
        x_next = x[:, 1:, :]
        x_prev_centered = x_prev - x_prev.mean(dim=(1, 2), keepdim=True)
        x_next_centered = x_next - x_next.mean(dim=(1, 2), keepdim=True)
        corr_num = (x_prev_centered * x_next_centered).mean(dim=(1, 2))
        corr_den = torch.sqrt(
            x_prev_centered.square().mean(dim=(1, 2))
            * x_next_centered.square().mean(dim=(1, 2))
            + 1e-6
        )
        return corr_num / corr_den

    def _extract_periodicity_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.size(1) <= 2:
            zeros = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            return zeros, zeros, zeros

        x_centered = (x - x.mean(dim=1, keepdim=True)).to(torch.float32)
        spectrum = torch.fft.rfft(x_centered, dim=1)
        power = spectrum.abs().square()[:, 1:, :]
        if power.size(1) == 0:
            zeros = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            return zeros, zeros, zeros

        power_avg = power.mean(dim=-1)
        total_power = power_avg.sum(dim=-1)
        dominant_power, dominant_idx = power_avg.max(dim=-1)
        dominant_period_proxy = 1.0 / (dominant_idx.to(torch.float32) + 1.0)
        spectral_concentration = dominant_power / (total_power + 1e-6)
        probability = power_avg / (total_power.unsqueeze(-1) + 1e-6)
        spectral_entropy = -(probability * (probability + 1e-8).log()).sum(dim=-1)
        if power_avg.size(-1) > 1:
            spectral_entropy = spectral_entropy / torch.log(
                torch.tensor(float(power_avg.size(-1)), device=x.device, dtype=torch.float32)
            )
        else:
            spectral_entropy = torch.zeros_like(dominant_period_proxy)
        return (
            dominant_period_proxy.to(x.dtype),
            spectral_concentration.to(x.dtype),
            spectral_entropy.to(x.dtype),
        )

    def _compute_query_key_from_series(self, x: torch.Tensor) -> torch.Tensor:
        level_series = x.mean(dim=2)
        volatility_series = x.std(dim=2, unbiased=False)
        if x.size(1) > 1:
            diff_mean = x[:, 1:, :].mean(dim=2) - x[:, :-1, :].mean(dim=2)
            diff_series = F.pad(diff_mean.abs(), (1, 0))
        else:
            diff_series = torch.zeros_like(level_series)

        signature = torch.cat(
            [
                self._downsample_series(level_series),
                self._downsample_series(volatility_series),
                self._downsample_series(diff_series),
            ],
            dim=-1,
        )
        return F.normalize(self.series_key_proj(signature.to(torch.float32)), dim=-1)

    def _compute_query_key_from_features(self, features: torch.Tensor) -> torch.Tensor:
        mean_feat = features.mean(dim=1)
        std_feat = features.std(dim=1, unbiased=False)
        max_feat = features.amax(dim=1)
        signature = torch.cat([mean_feat, std_feat, max_feat], dim=-1)
        return F.normalize(self.feature_key_proj(signature.to(torch.float32)), dim=-1)

    def _compute_query_key(
        self,
        x: torch.Tensor,
        retrieval_query_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if torch.is_tensor(retrieval_query_features) and retrieval_query_features.dim() == 3:
            return self._compute_query_key_from_features(retrieval_query_features)
        return self._compute_query_key_from_series(x)

    def _compute_stats(self, x: torch.Tensor) -> torch.Tensor:
        sample_mean = x.mean(dim=(1, 2))
        sample_std = x.std(dim=(1, 2), unbiased=False)
        trend = x[:, -1, :].mean(dim=1) - x[:, 0, :].mean(dim=1)
        if x.size(1) > 1:
            first_diff = x[:, 1:, :] - x[:, :-1, :]
            volatility = first_diff.abs().mean(dim=(1, 2))
            lag1_corr = self._lag1_correlation(x)
        else:
            volatility = torch.zeros_like(sample_mean)
            lag1_corr = torch.zeros_like(sample_mean)
        peak_valley = x.amax(dim=(1, 2)) - x.amin(dim=(1, 2))
        dominant_period_proxy, spectral_concentration, spectral_entropy = self._extract_periodicity_stats(x)
        regime_energy = x.square().mean(dim=(1, 2)).sqrt()
        return torch.stack(
            [
                sample_mean,
                sample_std,
                trend,
                volatility,
                peak_valley,
                lag1_corr,
                dominant_period_proxy,
                spectral_concentration,
                spectral_entropy,
                regime_energy,
                x[:, -1, :].std(dim=1, unbiased=False),
                x[:, 0, :].std(dim=1, unbiased=False),
            ],
            dim=-1,
        )

    def _compute_sensor_context(self, x: torch.Tensor) -> torch.Tensor:
        mean_feat = x.mean(dim=1)
        std_feat = x.std(dim=1, unbiased=False)
        trend_feat = x[:, -1, :] - x[:, 0, :]
        if x.size(1) > 1:
            diff_feat = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=1)
        else:
            diff_feat = torch.zeros_like(mean_feat)
        sensor_signature = torch.stack([mean_feat, std_feat, trend_feat, diff_feat], dim=-1)
        return self.sensor_value_proj(sensor_signature.to(torch.float32)).to(x.dtype)

    def _build_empty_bundle(self, route_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = route_features.size(0)
        device = route_features.device
        dtype = route_features.dtype
        return {
            "retrieval_context": torch.zeros_like(route_features),
            "retrieval_confidence": torch.zeros(batch_size, device=device, dtype=dtype),
            "retrieval_similarity": torch.zeros(batch_size, device=device, dtype=dtype),
            "retrieval_stats": torch.zeros(batch_size, self.RETRIEVAL_STATS_DIM, device=device, dtype=dtype),
            "retrieval_topk_indices": torch.full((batch_size, self.top_k), -1, device=device, dtype=torch.long),
            "retrieval_topk_scores": torch.zeros(batch_size, self.top_k, device=device, dtype=dtype),
            "retrieval_topk_lags": torch.zeros(batch_size, self.top_k, device=device, dtype=dtype),
            "retrieval_agreement": torch.zeros(batch_size, device=device, dtype=dtype),
            "retrieval_gap": torch.zeros(batch_size, device=device, dtype=dtype),
            "retrieval_slot_count": torch.zeros(batch_size, device=device, dtype=dtype),
            "retrieval_evidence_tokens": torch.zeros(batch_size, 0, self.d_model, device=device, dtype=dtype),
            "retrieval_evidence_summary": torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
            "retrieval_regime_summary": torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
            "retrieval_best_lag": torch.zeros(batch_size, device=device, dtype=dtype),
        }

    @torch.no_grad()
    def reset_memory(self) -> None:
        self.prototype_keys.zero_()
        self.prototype_values.zero_()
        self.prototype_stats.zero_()
        self.prototype_valid_mask.zero_()
        self.prototype_write_ptr.zero_()

    @torch.no_grad()
    def _update_memory(
        self,
        query_keys: torch.Tensor,
        route_features: torch.Tensor,
        stats: torch.Tensor,
        sensor_mask: torch.Tensor,
    ) -> None:
        batch_size, num_vars, _ = route_features.shape
        valid_vars = min(num_vars, self.max_vars)
        for batch_idx in range(batch_size):
            write_idx = int(self.prototype_write_ptr.item())
            self.prototype_keys[write_idx].copy_(query_keys[batch_idx])
            self.prototype_values[write_idx].zero_()
            value = route_features[batch_idx, :valid_vars].detach()
            mask = sensor_mask[batch_idx, :valid_vars].to(value.dtype).unsqueeze(-1)
            self.prototype_values[write_idx, :valid_vars].copy_(value * mask)
            self.prototype_stats[write_idx].copy_(stats[batch_idx].detach())
            self.prototype_valid_mask[write_idx] = True
            self.prototype_write_ptr[0] = (write_idx + 1) % self.num_slots

    def _masked_topk_weights(
        self,
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masked_scores = scores.masked_fill(~valid_mask, -1e4)
        top_scores, top_indices = torch.topk(masked_scores, k=top_k, dim=-1)
        top_valid = top_scores > -1e3
        safe_scores = top_scores.masked_fill(~top_valid, -1e4)
        weights = torch.softmax(safe_scores / self.confidence_temperature, dim=-1)
        weights = weights * top_valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return top_scores, top_indices, weights

    def _build_evidence_tokens(
        self,
        gathered_context: torch.Tensor,
        gathered_stats: torch.Tensor,
        top_scores: torch.Tensor,
        lag_values: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_tokens = gathered_context.mean(dim=2)
        stats_tokens = self.stats_value_proj(gathered_stats.to(torch.float32)).to(context_tokens.dtype)
        numeric_features = torch.stack(
            [
                top_scores.to(context_tokens.dtype),
                top_scores.abs().to(context_tokens.dtype),
                lag_values.to(context_tokens.dtype),
                lag_values.abs().to(context_tokens.dtype),
            ],
            dim=-1,
        )
        numeric_tokens = self.numeric_value_proj(numeric_features.to(torch.float32)).to(context_tokens.dtype)
        evidence_tokens = self.evidence_norm(context_tokens + stats_tokens + numeric_tokens)
        evidence_summary = torch.sum(evidence_tokens * weights.unsqueeze(-1), dim=1)
        return evidence_tokens, evidence_summary

    def _finalize_bundle(
        self,
        route_features: torch.Tensor,
        gathered_context: torch.Tensor,
        gathered_stats: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        weights: torch.Tensor,
        lag_values: Optional[torch.Tensor],
        slot_count: float,
    ) -> Dict[str, torch.Tensor]:
        batch_size = route_features.size(0)
        dtype = route_features.dtype
        device = route_features.device
        if lag_values is None:
            lag_values = torch.zeros(batch_size, top_scores.size(1), device=device, dtype=dtype)

        retrieved_context = torch.sum(gathered_context * weights.unsqueeze(-1).unsqueeze(-1), dim=1)
        retrieved_stats = torch.sum(gathered_stats * weights.unsqueeze(-1), dim=1)
        evidence_tokens, evidence_summary = self._build_evidence_tokens(
            gathered_context=gathered_context,
            gathered_stats=gathered_stats,
            top_scores=top_scores,
            lag_values=lag_values,
            weights=weights,
        )

        retrieval_similarity = (weights * top_scores.to(weights.dtype)).sum(dim=-1)
        retrieval_confidence = torch.sigmoid(top_scores[:, 0].to(dtype) + self.confidence_bias)
        current_pool = route_features.mean(dim=1)
        retrieved_pool = retrieved_context.mean(dim=1)
        retrieval_agreement = F.cosine_similarity(current_pool, retrieved_pool, dim=-1, eps=1e-6)
        retrieval_gap = (current_pool - retrieved_pool).square().mean(dim=-1).sqrt()
        retrieval_gap = retrieval_gap / (
            current_pool.square().mean(dim=-1).sqrt()
            + retrieved_pool.square().mean(dim=-1).sqrt()
            + 1e-6
        )

        return {
            "retrieval_context": retrieved_context,
            "retrieval_confidence": retrieval_confidence,
            "retrieval_similarity": retrieval_similarity.to(dtype),
            "retrieval_stats": retrieved_stats.to(dtype),
            "retrieval_topk_indices": top_indices,
            "retrieval_topk_scores": top_scores.to(dtype),
            "retrieval_topk_lags": lag_values.to(dtype),
            "retrieval_agreement": retrieval_agreement.to(dtype),
            "retrieval_gap": retrieval_gap.to(dtype),
            "retrieval_slot_count": torch.full((batch_size,), float(slot_count), device=device, dtype=dtype),
            "retrieval_evidence_tokens": evidence_tokens.to(dtype),
            "retrieval_evidence_summary": evidence_summary.to(dtype),
            "retrieval_regime_summary": evidence_summary.to(dtype),
            "retrieval_best_lag": lag_values[:, 0].to(dtype),
        }

    def _retrieve_from_candidates(
        self,
        x: torch.Tensor,
        route_features: torch.Tensor,
        retrieval_query_features: Optional[torch.Tensor],
        retrieval_x: torch.Tensor,
        retrieval_mask: Optional[torch.Tensor],
        retrieval_lags: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_candidates, seq_len, num_vars = retrieval_x.shape
        bundle = self._build_empty_bundle(route_features)
        if num_candidates == 0:
            return bundle

        if retrieval_mask is None:
            retrieval_mask = torch.ones(batch_size, num_candidates, device=x.device, dtype=torch.bool)
        else:
            retrieval_mask = retrieval_mask.to(device=x.device, dtype=torch.bool)

        flat_candidates = retrieval_x.reshape(batch_size * num_candidates, seq_len, num_vars)
        candidate_keys = self._compute_query_key_from_series(flat_candidates).reshape(batch_size, num_candidates, self.key_dim)
        candidate_stats = self._compute_stats(flat_candidates).reshape(batch_size, num_candidates, self.RETRIEVAL_STATS_DIM).to(route_features.dtype)
        candidate_context = self._compute_sensor_context(flat_candidates).reshape(batch_size, num_candidates, num_vars, self.d_model).to(route_features.dtype)

        query_keys = self._compute_query_key(x, retrieval_query_features)
        scores = torch.einsum('bd,bkd->bk', query_keys, candidate_keys)
        top_k = min(self.top_k, num_candidates)
        top_scores, top_indices, weights = self._masked_topk_weights(scores, retrieval_mask, top_k)

        gathered_context = torch.gather(
            candidate_context,
            1,
            top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_vars, self.d_model),
        )
        gathered_stats = torch.gather(
            candidate_stats,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.RETRIEVAL_STATS_DIM),
        )

        lag_values = None
        if torch.is_tensor(retrieval_lags):
            lag_tensor = retrieval_lags.to(device=x.device, dtype=route_features.dtype)
            if lag_tensor.dim() == 1:
                lag_tensor = lag_tensor.unsqueeze(0).expand(batch_size, -1)
            if lag_tensor.dim() >= 2:
                lag_values = torch.gather(lag_tensor, 1, top_indices)

        bundle = self._finalize_bundle(
            route_features=route_features,
            gathered_context=gathered_context,
            gathered_stats=gathered_stats,
            top_scores=top_scores,
            top_indices=top_indices,
            weights=weights,
            lag_values=lag_values,
            slot_count=float(num_candidates),
        )
        return bundle

    def _retrieve_from_memory(
        self,
        x: torch.Tensor,
        route_features: torch.Tensor,
        retrieval_query_features: Optional[torch.Tensor],
        sensor_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bundle = self._build_empty_bundle(route_features)
        valid_indices = torch.nonzero(self.prototype_valid_mask, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return bundle

        keys = F.normalize(self.prototype_keys[valid_indices].to(route_features.dtype), dim=-1)
        query_keys = self._compute_query_key(x, retrieval_query_features).to(route_features.dtype)
        scores = torch.matmul(query_keys, keys.transpose(0, 1))
        top_k = min(self.top_k, scores.size(-1))
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
        top_scores, top_positions, weights = self._masked_topk_weights(scores, valid_mask, top_k)
        top_indices = valid_indices[top_positions]

        gathered_context = self.prototype_values[top_indices, : route_features.size(1), :].to(route_features.dtype)
        gathered_context = gathered_context * sensor_mask.unsqueeze(1).unsqueeze(-1).to(route_features.dtype)
        gathered_stats = self.prototype_stats[top_indices].to(route_features.dtype)

        return self._finalize_bundle(
            route_features=route_features,
            gathered_context=gathered_context,
            gathered_stats=gathered_stats,
            top_scores=top_scores,
            top_indices=top_indices,
            weights=weights,
            lag_values=None,
            slot_count=float(valid_indices.numel()),
        )

    def forward(
        self,
        x: torch.Tensor,
        route_features: torch.Tensor,
        retrieval_query_features: Optional[torch.Tensor] = None,
        time_state: Optional[dict] = None,
        segment_summaries: Optional[torch.Tensor] = None,
        local_tokens: Optional[torch.Tensor] = None,
        global_temporal_memory: Optional[torch.Tensor] = None,
        retrieval_x: Optional[torch.Tensor] = None,
        sensor_mask: Optional[torch.Tensor] = None,
        retrieval_mask: Optional[torch.Tensor] = None,
        retrieval_lags: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _ = time_state
        _ = segment_summaries
        _ = local_tokens
        _ = global_temporal_memory

        if route_features.dim() != 3:
            raise ValueError("route_features must have shape [B, V, D]")
        batch_size, num_vars, _ = route_features.shape
        if sensor_mask is None:
            sensor_mask = torch.ones(batch_size, num_vars, device=x.device, dtype=torch.bool)
        else:
            sensor_mask = sensor_mask.to(device=x.device, dtype=torch.bool)

        if torch.is_tensor(retrieval_x) and retrieval_x.dim() == 4:
            return self._retrieve_from_candidates(
                x=x,
                route_features=route_features,
                retrieval_query_features=retrieval_query_features,
                retrieval_x=retrieval_x.to(device=x.device, dtype=x.dtype),
                retrieval_mask=retrieval_mask,
                retrieval_lags=retrieval_lags,
            )

        bundle = self._retrieve_from_memory(
            x=x,
            route_features=route_features,
            retrieval_query_features=retrieval_query_features,
            sensor_mask=sensor_mask,
        )
        if self.training:
            query_keys = self._compute_query_key(x, retrieval_query_features)
            stats = self._compute_stats(x).to(route_features.dtype)
            self._update_memory(query_keys.detach(), route_features.detach(), stats.detach(), sensor_mask.detach())
        return bundle


PeriodicRegimeRetriever = TemporalPatternRetriever
