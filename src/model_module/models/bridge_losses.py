"""semantic bridge 的独立监督损失。 

该模块用于 bridge-only 阶段，把 `semantic_bridge` 的输出约束到更稳定的时序/关系/证据语义上，
避免一开始就把 bridge 直接暴露给 LLM 主路径。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    """返回与 reference 同设备的标量零损失。"""

    return reference.new_zeros(())


def _pooled_target(target: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """把不同形状的目标统一压成 `[B, D]`。"""

    if not torch.is_tensor(target):
        return None
    if target.dim() == 3:
        return target.mean(dim=1)
    if target.dim() == 2:
        return target
    return None


def _resolve_evidence_target(
    retrieval_evidence: Optional[dict],
    reference: torch.Tensor,
) -> Optional[torch.Tensor]:
    """优先取 retrieval_evidence_summary，其次退回 pooled retrieval_context。"""

    if not isinstance(retrieval_evidence, dict):
        return None

    evidence_summary = retrieval_evidence.get("retrieval_evidence_summary")
    pooled_summary = _pooled_target(evidence_summary)
    if pooled_summary is not None:
        return pooled_summary.to(device=reference.device, dtype=reference.dtype)

    retrieval_context = retrieval_evidence.get("retrieval_context")
    if torch.is_tensor(retrieval_context):
        if retrieval_context.dim() == 3:
            return retrieval_context.mean(dim=1).to(device=reference.device, dtype=reference.dtype)
        if retrieval_context.dim() == 2:
            return retrieval_context.to(device=reference.device, dtype=reference.dtype)

    return None


def _mse_plus_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """用 MSE + cosine distance 约束语义对齐。"""

    target = target.detach()
    mse = F.mse_loss(pred, target)
    cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1e-6).mean()
    return mse + cosine


def bridge_alignment_loss(
    bridge_state: Optional[dict],
    sensor_state: torch.Tensor,
    time_summary: torch.Tensor,
    retrieval_evidence: Optional[dict],
) -> torch.Tensor:
    """约束 bridge_summary 对齐多路语义目标的平均语义。"""

    if not isinstance(bridge_state, dict):
        return _zero_like(sensor_state)

    bridge_summary = _pooled_target(bridge_state.get("bridge_summary"))
    if bridge_summary is None:
        return _zero_like(sensor_state)

    targets = [time_summary.mean(dim=1).detach(), sensor_state.mean(dim=1).detach()]
    evidence_target = _resolve_evidence_target(retrieval_evidence, sensor_state)
    if evidence_target is not None:
        targets.append(evidence_target.detach())

    target_summary = torch.stack(targets, dim=0).mean(dim=0)
    return _mse_plus_cosine(bridge_summary, target_summary)


def token_group_consistency_loss(
    bridge_state: Optional[dict],
    sensor_state: torch.Tensor,
    time_summary: torch.Tensor,
    retrieval_evidence: Optional[dict],
) -> torch.Tensor:
    """分别约束时间/关系/证据 token group 与对应目标的一致性。"""

    if not isinstance(bridge_state, dict):
        return _zero_like(sensor_state)

    losses = []

    time_tokens = _pooled_target(bridge_state.get("time_event_tokens"))
    if time_tokens is not None:
        losses.append(_mse_plus_cosine(time_tokens, time_summary.mean(dim=1)))

    relation_tokens = _pooled_target(bridge_state.get("relation_tokens"))
    if relation_tokens is not None:
        losses.append(_mse_plus_cosine(relation_tokens, sensor_state.mean(dim=1)))

    evidence_tokens = _pooled_target(bridge_state.get("evidence_tokens"))
    evidence_target = _resolve_evidence_target(retrieval_evidence, sensor_state)
    if evidence_tokens is not None and evidence_target is not None:
        losses.append(_mse_plus_cosine(evidence_tokens, evidence_target))

    if len(losses) == 0:
        return _zero_like(sensor_state)
    return torch.stack(losses).mean()


def evidence_bridge_agreement_loss(
    bridge_state: Optional[dict],
    retrieval_evidence: Optional[dict],
    sensor_state: torch.Tensor,
) -> torch.Tensor:
    """约束 bridge_summary 与 retrieval evidence summary 保持一致。"""

    if not isinstance(bridge_state, dict):
        return _zero_like(sensor_state)

    bridge_summary = _pooled_target(bridge_state.get("bridge_summary"))
    evidence_target = _resolve_evidence_target(retrieval_evidence, sensor_state)
    if bridge_summary is None or evidence_target is None:
        return _zero_like(sensor_state)
    return _mse_plus_cosine(bridge_summary, evidence_target)
