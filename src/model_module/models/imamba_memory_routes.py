"""iMambaMemory 路由构建与辅助函数。

该模块负责初始化 temporal route、retrieval 模块以及 embedding 输出拆包逻辑，
用于把主编码器的结构性分支配置从主类中解耦出来。
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    from src.model_module.layers.temporal_detail import TemporalDetailRoute
except ImportError:
    TemporalDetailRoute = None

try:
    from src.model_module.layers.temporal_retrieval import PeriodicRegimeRetriever
except ImportError:
    PeriodicRegimeRetriever = None


def build_temporal_detail_route(model: Any, cfg) -> Optional[nn.Module]:
    """按配置构造时间细节分支。"""

    if not getattr(cfg.model, "use_temporal_detail_route", False):
        return None
    if TemporalDetailRoute is None:
        print("Warning: TemporalDetailRoute is enabled in config but not available. Disabling route.")
        return None

    temporal_max_tokens = getattr(cfg.model, "temporal_max_patches", 128)
    if temporal_max_tokens is None or temporal_max_tokens <= 0:
        temporal_max_tokens = max(model.seq_len, 512)

    route_kwargs = {
        "d_model": model.d_model,
        "patch_len": getattr(cfg.model, "temporal_patch_len", 16),
        "stride": getattr(cfg.model, "temporal_stride", 8),
        "hidden_channels": getattr(cfg.model, "temporal_hidden_dim", 32),
        "dropout": getattr(cfg.model, "temporal_route_dropout", 0.0),
        "num_heads": getattr(cfg.model, "temporal_num_heads", 4),
        "num_layers": getattr(cfg.model, "temporal_num_layers", 1),
        "max_tokens": temporal_max_tokens,
        "state_fusion_mode": getattr(cfg.model, "temporal_state_fusion_mode", "all"),
    }

    signature = inspect.signature(TemporalDetailRoute.__init__)
    supported = set(signature.parameters.keys())
    filtered_kwargs = {key: value for key, value in route_kwargs.items() if key in supported}
    return TemporalDetailRoute(**filtered_kwargs)


def build_temporal_retrieval(model: Any, cfg) -> Optional[nn.Module]:
    """按配置构造 retrieval evidence 模块。"""

    if not getattr(cfg.model, "use_temporal_retrieval", False):
        return None
    if PeriodicRegimeRetriever is None:
        print("Warning: PeriodicRegimeRetriever is enabled in config but not available. Disabling retrieval.")
        return None

    retrieval_kwargs = {
        "d_model": model.d_model,
        "num_slots": getattr(cfg.model, "retrieval_num_slots", 128),
        "top_k": getattr(cfg.model, "retrieval_top_k", 4),
        "key_dim": getattr(cfg.model, "retrieval_key_dim", model.d_model),
        "key_downsample_len": getattr(cfg.model, "retrieval_key_downsample_len", 128),
        "max_vars": getattr(cfg.model, "max_sensor_id", model.max_sensor_id),
        "confidence_temperature": getattr(cfg.model, "retrieval_confidence_temperature", 1.0),
        "confidence_bias": getattr(cfg.model, "retrieval_confidence_bias", 0.0),
    }
    signature = inspect.signature(PeriodicRegimeRetriever.__init__)
    supported = set(signature.parameters.keys())
    filtered_kwargs = {key: value for key, value in retrieval_kwargs.items() if key in supported}
    return PeriodicRegimeRetriever(**filtered_kwargs)


def apply_temporal_bridge(
    model: Any,
    base_features: torch.Tensor,
    temporal_tokens: Optional[torch.Tensor],
    temporal_features: torch.Tensor,
) -> torch.Tensor:
    """利用 token 级注意力把主路特征与 temporal token 对齐。"""

    if model.temporal_bridge_attn is None or temporal_tokens is None:
        return temporal_features

    batch_size, num_vars, hidden_dim = base_features.shape
    token_count = temporal_tokens.size(2)
    mode = str(getattr(model, "temporal_bridge_mode", "cross_attn")).lower()

    if mode == "cross_attn":
        query = model.temporal_bridge_query_norm(base_features.reshape(batch_size * num_vars, 1, hidden_dim))
        token_bank = model.temporal_bridge_token_norm(
            temporal_tokens.reshape(batch_size * num_vars, token_count, hidden_dim)
        )
        attn_out, _ = model.temporal_bridge_attn(query, token_bank, token_bank, need_weights=False)
        attn_out = attn_out.reshape(batch_size, num_vars, hidden_dim)
    elif mode in {"global_cross_attn", "lag_aware_cross_attn"}:
        query = model.temporal_bridge_query_norm(base_features)
        token_bank = model.temporal_bridge_token_norm(
            temporal_tokens.reshape(batch_size, num_vars * token_count, hidden_dim)
        )
        if mode == "lag_aware_cross_attn" and model.temporal_bridge_lag_proj is not None:
            lag_positions = torch.linspace(
                -1.0,
                0.0,
                steps=token_count,
                device=temporal_tokens.device,
                dtype=temporal_tokens.dtype,
            ).view(1, 1, token_count, 1)
            lag_bias = model.temporal_bridge_lag_proj(lag_positions)
            lag_bias = lag_bias.expand(batch_size, num_vars, token_count, hidden_dim)
            token_bank = token_bank + lag_bias.reshape(batch_size, num_vars * token_count, hidden_dim)
        attn_out, _ = model.temporal_bridge_attn(query, token_bank, token_bank, need_weights=False)
    elif mode == "grouped_cross_attn":
        query = model.temporal_bridge_query_norm(base_features)
        norm_tokens = model.temporal_bridge_token_norm(temporal_tokens)
        group_size = max(1, int(getattr(model, "temporal_bridge_group_size", 4)))
        group_outputs = []
        for start in range(0, num_vars, group_size):
            end = min(num_vars, start + group_size)
            group_query = query[:, start:end, :]
            group_tokens = norm_tokens[:, start:end, :, :].reshape(
                batch_size,
                (end - start) * token_count,
                hidden_dim,
            )
            group_out, _ = model.temporal_bridge_attn(
                group_query,
                group_tokens,
                group_tokens,
                need_weights=False,
            )
            group_outputs.append(group_out)
        attn_out = torch.cat(group_outputs, dim=1)
    else:
        raise ValueError(f"Unsupported temporal_bridge_mode={mode}")

    bridged_features = 0.5 * (temporal_features + attn_out)
    return model.temporal_bridge_out_norm(bridged_features)


def select_temporal_bridge_tokens(
    model: Any,
    time_state: Dict[str, Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    """按配置选择 bridge 使用的层级 token bank。"""

    source = str(getattr(model, "temporal_bridge_token_source", "local")).lower()
    source_map = {
        "local": ("local_tokens",),
        "segment": ("segment_summaries",),
        "global": ("global_temporal_memory",),
        "summary": ("summary",),
        "local_segment": ("local_tokens", "segment_summaries"),
        "local_global": ("local_tokens", "global_temporal_memory"),
        "segment_global": ("segment_summaries", "global_temporal_memory"),
        "all": ("local_tokens", "segment_summaries", "global_temporal_memory"),
    }
    if source not in source_map:
        raise ValueError(
            f"Unsupported temporal_bridge_token_source={source}. "
            f"Expected one of {sorted(source_map)}"
        )

    tokens: list[torch.Tensor] = []
    for key in source_map[source]:
        candidate = time_state.get(key)
        if not torch.is_tensor(candidate):
            continue
        if candidate.dim() == 3:
            candidate = candidate.unsqueeze(2)
        if candidate.dim() == 4 and candidate.size(2) > 0:
            tokens.append(candidate)

    if not tokens:
        return None
    return torch.cat(tokens, dim=2)


def _valid_summary_shape(summary: torch.Tensor, reference: torch.Tensor) -> bool:
    """检查 summary 是否与主路特征形状一致。"""

    return (
        summary.dim() == reference.dim()
        and summary.size(0) == reference.size(0)
        and summary.size(1) == reference.size(1)
        and summary.size(2) == reference.size(2)
    )


def build_time_state_from_route_output(
    route_output: Any,
    default_summary: torch.Tensor,
) -> Dict[str, Optional[torch.Tensor]]:
    """把 temporal route 输出规范化为统一的 time_state。"""

    summary = default_summary
    local_tokens: Optional[torch.Tensor] = None
    segment_summaries: Optional[torch.Tensor] = None
    global_temporal_memory: Optional[torch.Tensor] = None

    if isinstance(route_output, dict):
        summary_candidate = route_output.get("summary")
        if torch.is_tensor(summary_candidate) and _valid_summary_shape(summary_candidate, default_summary):
            summary = summary_candidate
        else:
            features_candidate = route_output.get("features")
            if torch.is_tensor(features_candidate) and _valid_summary_shape(features_candidate, default_summary):
                summary = features_candidate

        token_candidate = route_output.get("local_tokens")
        if not torch.is_tensor(token_candidate):
            token_candidate = route_output.get("tokens")
        if torch.is_tensor(token_candidate):
            local_tokens = token_candidate

        segment_candidate = route_output.get("segment_summaries")
        if torch.is_tensor(segment_candidate):
            segment_summaries = segment_candidate

        global_candidate = route_output.get("global_temporal_memory")
        if torch.is_tensor(global_candidate):
            global_temporal_memory = global_candidate
    elif torch.is_tensor(route_output):
        if _valid_summary_shape(route_output, default_summary):
            summary = route_output

    return {
        "local_tokens": local_tokens,
        "segment_summaries": segment_summaries,
        "global_temporal_memory": global_temporal_memory,
        "summary": summary,
    }


def select_retrieval_query_features(
    time_state: Dict[str, Optional[torch.Tensor]],
    fallback_features: torch.Tensor,
) -> Tuple[torch.Tensor, str]:
    """优先从 time_state 选择 retrieval 查询特征。"""

    segment_summaries = time_state.get("segment_summaries")
    if torch.is_tensor(segment_summaries) and segment_summaries.dim() == 4:
        return segment_summaries.mean(dim=2), "segment_summaries"

    local_tokens = time_state.get("local_tokens")
    if torch.is_tensor(local_tokens) and local_tokens.dim() == 4:
        return local_tokens.mean(dim=2), "local_tokens"

    summary = time_state.get("summary")
    if torch.is_tensor(summary) and _valid_summary_shape(summary, fallback_features):
        return summary, "time_summary"

    return fallback_features, "primary_features"


def run_temporal_retrieval(
    retrieval_module: Optional[nn.Module],
    x: torch.Tensor,
    route_features: torch.Tensor,
    time_state: Dict[str, Optional[torch.Tensor]],
    retrieval_x: Optional[torch.Tensor],
    sensor_mask: Optional[torch.Tensor],
    retrieval_mask: Optional[torch.Tensor],
    retrieval_lags: Optional[torch.Tensor],
) -> Optional[dict]:
    """兼容不同 forward 签名调用 retrieval，并优先传递 time_state。"""

    if retrieval_module is None:
        return None

    retrieval_query_features, query_source = select_retrieval_query_features(
        time_state=time_state,
        fallback_features=route_features,
    )

    call_kwargs: Dict[str, Any] = {
        "route_features": route_features,
        "retrieval_query_features": retrieval_query_features,
        "time_state": time_state,
        "segment_summaries": time_state.get("segment_summaries"),
        "local_tokens": time_state.get("local_tokens"),
        "global_temporal_memory": time_state.get("global_temporal_memory"),
        "retrieval_x": retrieval_x,
        "sensor_mask": sensor_mask,
        "retrieval_mask": retrieval_mask,
        "retrieval_lags": retrieval_lags,
    }

    signature = inspect.signature(retrieval_module.forward)
    supports_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if not supports_var_kwargs:
        supported = set(signature.parameters.keys())
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in supported}

    retrieval_bundle = retrieval_module(x, **call_kwargs)
    if isinstance(retrieval_bundle, dict):
        retrieval_bundle.setdefault("retrieval_query_source", query_source)
    return retrieval_bundle


def apply_sensor_id_embedding(model: Any, embeds: torch.Tensor) -> torch.Tensor:
    """在变量 token 上叠加显式传感器身份嵌入。"""

    if model.sensor_id_embedding is None:
        return embeds
    num_vars = embeds.size(1)
    max_idx = model.sensor_id_embedding.num_embeddings - 1
    sensor_ids = torch.arange(num_vars, device=embeds.device).clamp(max=max_idx)
    sensor_id_embeds = model.sensor_id_embedding(sensor_ids).unsqueeze(0).to(embeds.dtype)
    return embeds + sensor_id_embeds


def unpack_embedding_output(
    embedding_output: Any,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
    """兼容多种 embedding 返回格式，并统一为三元组输出。"""

    if isinstance(embedding_output, dict):
        embeds = embedding_output.get("embeds") or embedding_output.get("features")
        if embeds is None:
            raise KeyError("Embedding dict output must contain 'embeds' or 'features'.")
        mask = embedding_output.get("mask", embedding_output.get("sensor_mask"))
        main_route_aux = embedding_output.get("main_route_aux")
    elif isinstance(embedding_output, (tuple, list)):
        if len(embedding_output) == 2:
            embeds, mask = embedding_output
            main_route_aux = None
        elif len(embedding_output) == 3:
            embeds, mask, main_route_aux = embedding_output
        else:
            raise ValueError("Embedding tuple/list output must have length 2 or 3.")
    else:
        raise TypeError(f"Unsupported embedding output type: {type(embedding_output)}")

    if mask is None:
        mask = torch.ones(x.size(0), x.size(2), device=x.device, dtype=torch.bool)
    if main_route_aux is not None and not isinstance(main_route_aux, dict):
        main_route_aux = {"aux": main_route_aux}
    return embeds, mask, main_route_aux


def main_embedding_forward(
    model: Any,
    x: torch.Tensor,
    sensor_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
    """执行主 embedding 路径，并兼容 raw/patch/decomposed 多种模式。"""

    if model.main_embedding_mode == "decomposed":
        if model.inverted_embed is None:
            raise RuntimeError("Decomposed embedding module is not initialized.")
        decomp_kwargs: Dict[str, Any] = {}
        try:
            forward_sig = inspect.signature(model.inverted_embed.forward)
            if "return_component_bundle" in forward_sig.parameters:
                decomp_kwargs["return_component_bundle"] = True
        except (TypeError, ValueError):
            pass
        embeds, mask, main_route_aux = unpack_embedding_output(
            model.inverted_embed(x, sensor_mask, **decomp_kwargs),
            x,
        )
        if main_route_aux is not None:
            main_route_aux = dict(main_route_aux)
            main_route_aux.setdefault("main_embedding_mode", model.main_embedding_mode)
        return apply_sensor_id_embedding(model, embeds), mask, main_route_aux

    if model.inverted_embed is not None:
        embeds, mask, main_route_aux = unpack_embedding_output(model.inverted_embed(x, sensor_mask), x)
        return apply_sensor_id_embedding(model, embeds), mask, main_route_aux

    if model.raw_inverted_embed is None or model.patch_inverted_embed is None or model.raw_patch_mix_logit is None:
        raise RuntimeError("Main embedding modules are not initialized correctly for raw_patch_mix mode.")

    raw_embeds, raw_mask, raw_aux = unpack_embedding_output(model.raw_inverted_embed(x, sensor_mask), x)
    patch_embeds, _, patch_aux = unpack_embedding_output(model.patch_inverted_embed(x, sensor_mask), x)
    mix_weight = torch.sigmoid(model.raw_patch_mix_logit).to(raw_embeds.dtype)
    mixed_embeds = (1.0 - mix_weight) * raw_embeds + mix_weight * patch_embeds
    main_route_aux: Dict[str, torch.Tensor] = {"raw_patch_mix_weight": mix_weight.detach().view(1)}
    if raw_aux is not None:
        for key, value in raw_aux.items():
            if isinstance(value, torch.Tensor):
                main_route_aux[f"raw_{key}"] = value
    if patch_aux is not None:
        for key, value in patch_aux.items():
            if isinstance(value, torch.Tensor):
                main_route_aux[f"patch_{key}"] = value
    return apply_sensor_id_embedding(model, mixed_embeds), raw_mask, main_route_aux

