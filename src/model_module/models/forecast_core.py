"""Forecast Core.

该模块承接当前最有效的 forecasting 主链，目标是把数值预测主路径从
`IMM_LLMModel` 的大协调器中剥离出来。

设计原则：
1. 保持当前 v2 数值预测主链的稳定边界
2. 不直接负责 LLM、semantic bridge 或 anomaly reasoning
3. 允许后续在不破坏预测主链的情况下保留 Semantic Sidecar
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.model_module.models.forecast_heads import (
    EvidenceCorrectionHead,
    RelationCorrectionHead,
    SensorForecastHead,
    TimeForecastHead,
)


class ForecastCore(nn.Module):
    """封装 forecasting 主链，减少 `IMM_LLMModel` 的路径耦合。"""

    def __init__(self, cfg) -> None:
        super().__init__()
        d_model = cfg.model.d_model
        pred_len = getattr(cfg.dataset, "pred_len", 96)

        self.pred_len = pred_len
        self.forecast_anchor_mode = str(getattr(cfg.model, "forecast_anchor_mode", "time_primary")).lower()
        self.sensor_head_mode = str(getattr(cfg.model, "sensor_head_mode", "plain_linear")).lower()
        self.relation_use_time_summary = bool(getattr(cfg.model, "relation_use_time_summary", False))

        self.time_aux_correction_scale = float(getattr(cfg.model, "time_aux_correction_scale", 0.0))
        self.relation_correction_scale = float(getattr(cfg.model, "relation_correction_scale", 0.1))
        self.evidence_correction_scale = float(getattr(cfg.model, "evidence_correction_scale", 0.1))

        self.time_forecast_head = TimeForecastHead(
            d_model=d_model,
            pred_len=pred_len,
            fusion_mode=str(getattr(cfg.model, "time_head_fusion_mode", "learned")),
        )
        self.sensor_forecast_head = SensorForecastHead(
            d_model=d_model,
            pred_len=pred_len,
            head_mode=self.sensor_head_mode,
        )
        self.relation_correction_head = RelationCorrectionHead(
            d_model=d_model,
            pred_len=pred_len,
            relation_code_dim=int(getattr(cfg.model, "relation_code_dim", 32)),
            use_time_summary=self.relation_use_time_summary,
        )
        self.evidence_correction_head = EvidenceCorrectionHead(
            d_model=d_model,
            pred_len=pred_len,
            evidence_hidden_dim=int(getattr(cfg.model, "evidence_hidden_dim", 64)),
        )

    def forward(
        self,
        sensor_state: torch.Tensor,
        time_state: Dict[str, torch.Tensor],
        time_summary: torch.Tensor,
        retrieval_bundle: Optional[dict],
    ) -> Dict[str, torch.Tensor]:
        """输出主预测、中间基底和修正量。"""

        time_outputs = self.time_forecast_head(
            time_state=time_state,
            fallback_features=time_summary,
        )
        y_time_base = time_outputs["y_time_base"]

        y_sensor_base = self.sensor_forecast_head(sensor_state)

        if self.forecast_anchor_mode == "sensor_primary":
            base_predictions_norm = y_sensor_base + self.time_aux_correction_scale * y_time_base
        else:
            base_predictions_norm = y_time_base

        if self.relation_correction_scale == 0.0:
            delta_relation = torch.zeros_like(base_predictions_norm)
        else:
            delta_relation = self.relation_correction_head(
                sensor_state=sensor_state,
                time_summary=time_summary if self.relation_use_time_summary else None,
            )

        if self.evidence_correction_scale == 0.0:
            delta_evidence = torch.zeros_like(base_predictions_norm)
        else:
            delta_evidence = self.evidence_correction_head(
                retrieval_evidence=retrieval_bundle,
                reference_features=sensor_state,
            )

        predictions_norm = (
            base_predictions_norm
            + self.relation_correction_scale * delta_relation
            + self.evidence_correction_scale * delta_evidence
        )

        return {
            "time_outputs": time_outputs,
            "y_time_base": y_time_base,
            "y_sensor_base": y_sensor_base,
            "base_predictions_norm": base_predictions_norm,
            "delta_relation": delta_relation,
            "delta_evidence": delta_evidence,
            "predictions_norm": predictions_norm,
        }
