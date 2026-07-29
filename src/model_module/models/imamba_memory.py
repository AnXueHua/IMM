"""iMamba-Memory 主编码器。

该模块负责主 embedding、backbone、temporal route、retrieval route 和 NMM 的汇合，是时序表征学习的核心编码器。
"""

from __future__ import annotations

import inspect
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.model_module.layers import InvertedEmbedding, NeuralMemoryMatrix, OutputHead
from src.model_module.models import register_model
from src.model_module.models.imamba_memory_routes import (
    apply_sensor_id_embedding,
    apply_temporal_bridge,
    build_temporal_detail_route,
    build_temporal_retrieval,
    build_time_state_from_route_output,
    main_embedding_forward,
    run_temporal_retrieval,
    select_temporal_bridge_tokens,
)
from src.model_module.utils import get_backbone


@register_model("iMamba_Memory")
class iMambaMemoryModel(nn.Module):
    # 该模型负责时序编码主干：主 route、temporal route、retrieval route 都在这里汇合。
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.model.d_model
        self.num_classes = cfg.model.num_classes
        self.pool_type = cfg.model.pool_type

        try:
            self.seq_len = cfg.dataset.seq_len
            self.patch_len = getattr(cfg.dataset, "patch_len", 16)
            self.stride = getattr(cfg.dataset, "stride", 16)
            self.use_patching = getattr(cfg.dataset, "use_patching", True)
        except AttributeError:
            self.seq_len = 512
            self.patch_len = 16
            self.stride = 16
            self.use_patching = True

        self.main_embedding_mode = getattr(cfg.model, "main_embedding_mode", None)
        self.use_sensor_id_embedding = getattr(cfg.model, "use_sensor_id_embedding", False)
        self.main_patch_len = getattr(cfg.model, "main_patch_len", self.patch_len)
        self.main_stride = getattr(cfg.model, "main_stride", self.stride)
        self.raw_patch_mix_bias = getattr(cfg.model, "raw_patch_mix_bias", 0.0)
        self.max_sensor_id = getattr(cfg.model, "max_sensor_id", 512)
        self.decomp_trend_kernel = int(getattr(cfg.model, "decomp_trend_kernel", 25))
        self.decomp_seasonal_kernel = int(getattr(cfg.model, "decomp_seasonal_kernel", 7))
        self.temporal_route_role = str(getattr(cfg.model, "temporal_route_role", "auxiliary")).lower()
        self.temporal_fusion_stage = str(getattr(cfg.model, "temporal_fusion_stage", "post_backbone")).lower()
        self.temporal_primary_residual_scale = float(
            getattr(cfg.model, "temporal_primary_residual_scale", 0.2)
        )
        self.temporal_bridge_mode = str(getattr(cfg.model, "temporal_bridge_mode", "pooled")).lower()
        self.temporal_bridge_token_source = str(
            getattr(cfg.model, "temporal_bridge_token_source", "local")
        ).lower()
        self.temporal_bridge_group_size = int(getattr(cfg.model, "temporal_bridge_group_size", 4))
        self.temporal_bridge_num_heads = int(getattr(cfg.model, "temporal_bridge_num_heads", 4))
        self.temporal_bridge_dropout = float(getattr(cfg.model, "temporal_bridge_dropout", 0.1))

        if self.main_embedding_mode is None:
            self.main_embedding_mode = "patch" if self.use_patching else "raw"
        else:
            self.main_embedding_mode = str(self.main_embedding_mode).lower()

        if self.decomp_trend_kernel <= 0 or self.decomp_seasonal_kernel <= 0:
            raise ValueError("Decomposition kernel sizes must be positive")
        if self.temporal_route_role not in {"auxiliary", "co_primary", "time_primary"}:
            raise ValueError(
                "Unsupported temporal_route_role="
                f"{self.temporal_route_role}, expected one of [auxiliary, co_primary, time_primary]"
            )
        valid_bridge_modes = {
            "pooled",
            "cross_attn",
            "global_cross_attn",
            "grouped_cross_attn",
            "lag_aware_cross_attn",
        }
        if self.temporal_bridge_mode not in valid_bridge_modes:
            raise ValueError(
                "Unsupported temporal_bridge_mode="
                f"{self.temporal_bridge_mode}, expected one of {sorted(valid_bridge_modes)}"
            )
        if self.temporal_bridge_group_size <= 0:
            raise ValueError("temporal_bridge_group_size must be positive")
        if self.temporal_bridge_num_heads <= 0:
            raise ValueError("temporal_bridge_num_heads must be positive")
        if self.temporal_fusion_stage not in {"post_backbone", "pre_backbone", "adaptive_gate", "cross_attn_asym"}:
            raise ValueError(
                "Unsupported temporal_fusion_stage="
                f"{self.temporal_fusion_stage}, expected one of [post_backbone, pre_backbone, adaptive_gate, cross_attn_asym]"
            )

        valid_main_modes = {"raw", "patch", "raw_patch_mix", "decomposed"}
        if self.main_embedding_mode not in valid_main_modes:
            raise ValueError(
                f"Unsupported main_embedding_mode={self.main_embedding_mode}, expected one of {sorted(valid_main_modes)}"
            )

        from src.model_module.layers.embeddings import InvertedPatchEmbedding

        try:
            from src.model_module.layers.embeddings import DecomposedInvertedEmbedding, MixedInvertedEmbedding
        except ImportError:
            DecomposedInvertedEmbedding = None
            MixedInvertedEmbedding = None

        # 主 route 的 tokenization 入口由 main_embedding_mode 控制。
        self.inverted_embed: Optional[nn.Module] = None
        self.raw_inverted_embed: Optional[nn.Module] = None
        self.patch_inverted_embed: Optional[nn.Module] = None
        self.raw_patch_mix_logit: Optional[nn.Parameter] = None
        if self.main_embedding_mode == "raw":
            self.inverted_embed = InvertedEmbedding(seq_len=self.seq_len, d_model=self.d_model)
        elif self.main_embedding_mode == "patch":
            self.inverted_embed = InvertedPatchEmbedding(
                seq_len=self.seq_len,
                d_model=self.d_model,
                patch_len=self.main_patch_len,
                stride=self.main_stride,
            )
        elif self.main_embedding_mode == "raw_patch_mix":
            if MixedInvertedEmbedding is not None:
                mixed_kwargs = {
                    "seq_len": self.seq_len,
                    "d_model": self.d_model,
                    "patch_len": self.main_patch_len,
                    "stride": self.main_stride,
                    "raw_patch_mix_bias": self.raw_patch_mix_bias,
                }
                mixed_signature = inspect.signature(MixedInvertedEmbedding.__init__)
                mixed_supported = set(mixed_signature.parameters.keys())
                mixed_filtered_kwargs = {k: v for k, v in mixed_kwargs.items() if k in mixed_supported}
                self.inverted_embed = MixedInvertedEmbedding(**mixed_filtered_kwargs)
            else:
                self.raw_inverted_embed = InvertedEmbedding(seq_len=self.seq_len, d_model=self.d_model)
                self.patch_inverted_embed = InvertedPatchEmbedding(
                    seq_len=self.seq_len,
                    d_model=self.d_model,
                    patch_len=self.main_patch_len,
                    stride=self.main_stride,
                )
                self.raw_patch_mix_logit = nn.Parameter(torch.tensor(float(self.raw_patch_mix_bias)))
        else:
            if DecomposedInvertedEmbedding is None:
                raise RuntimeError("DecomposedInvertedEmbedding is not available")
            residual_patch_len = max(2, min(self.seq_len, max(2, self.main_patch_len // 2)))
            residual_stride = max(1, self.main_stride // 2)
            self.inverted_embed = DecomposedInvertedEmbedding(
                seq_len=self.seq_len,
                d_model=self.d_model,
                trend_kernel_size=self.decomp_trend_kernel,
                seasonal_kernel_size=self.decomp_seasonal_kernel,
                seasonal_patch_len=self.main_patch_len,
                seasonal_stride=self.main_stride,
                residual_mode="short_patch",
                residual_patch_len=residual_patch_len,
                residual_stride=residual_stride,
                use_sensor_id_embedding=False,
                max_vars=self.max_sensor_id,
                return_component_bundle=True,
            )

        self.sensor_id_embedding: Optional[nn.Embedding] = None
        if self.use_sensor_id_embedding:
            self.sensor_id_embedding = nn.Embedding(self.max_sensor_id, self.d_model)

        # backbone 负责主跨传感器建模，temporal route 和 retrieval route 作为补充信息源。
        self.backbone = get_backbone(cfg)
        self.nmm = NeuralMemoryMatrix(d_model=self.d_model, nmm_cfg=cfg.model.nmm_cfg)
        self.temporal_detail_route = build_temporal_detail_route(self, cfg)
        self.temporal_bridge_query_norm: Optional[nn.LayerNorm] = None
        self.temporal_bridge_token_norm: Optional[nn.LayerNorm] = None
        self.temporal_bridge_attn: Optional[nn.MultiheadAttention] = None
        self.temporal_bridge_out_norm: Optional[nn.LayerNorm] = None
        self.temporal_bridge_lag_proj: Optional[nn.Linear] = None
        if self.temporal_detail_route is not None and self.temporal_bridge_mode != "pooled":
            bridge_heads = min(self.temporal_bridge_num_heads, self.d_model)
            while bridge_heads > 1 and self.d_model % bridge_heads != 0:
                bridge_heads -= 1
            bridge_heads = max(1, bridge_heads)
            self.temporal_bridge_query_norm = nn.LayerNorm(self.d_model)
            self.temporal_bridge_token_norm = nn.LayerNorm(self.d_model)
            self.temporal_bridge_attn = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=bridge_heads,
                dropout=self.temporal_bridge_dropout,
                batch_first=True,
            )
            self.temporal_bridge_out_norm = nn.LayerNorm(self.d_model)
            if self.temporal_bridge_mode == "lag_aware_cross_attn":
                self.temporal_bridge_lag_proj = nn.Linear(1, self.d_model, bias=False)
        self.temporal_retrieval = build_temporal_retrieval(self, cfg)
        self.use_temporal_pre_backbone_fusion = (
            self.temporal_detail_route is not None
            and self.temporal_route_role in {"co_primary", "time_primary"}
            and self.temporal_fusion_stage in {"pre_backbone", "adaptive_gate", "cross_attn_asym"}
        )

        self.adaptive_gate: Optional[nn.Sequential] = None
        if self.temporal_fusion_stage == "adaptive_gate":
            self.adaptive_gate = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, 2),
                nn.Softmax(dim=-1),
            )

        self.cross_attn_asym: Optional[nn.MultiheadAttention] = None
        self.cross_attn_asym_q_norm: Optional[nn.LayerNorm] = None
        self.cross_attn_asym_kv_norm: Optional[nn.LayerNorm] = None
        self.cross_attn_asym_out_norm: Optional[nn.LayerNorm] = None
        if self.temporal_fusion_stage == "cross_attn_asym":
            # 与 temporal bridge 复用头数与 dropout 配置，保持行为一致。
            heads = min(self.temporal_bridge_num_heads, self.d_model)
            while heads > 1 and self.d_model % heads != 0:
                heads -= 1
            heads = max(1, heads)
            self.cross_attn_asym = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=heads,
                dropout=self.temporal_bridge_dropout,
                batch_first=True,
            )
            self.cross_attn_asym_q_norm = nn.LayerNorm(self.d_model)
            self.cross_attn_asym_kv_norm = nn.LayerNorm(self.d_model)
            self.cross_attn_asym_out_norm = nn.LayerNorm(self.d_model)

        self.head = OutputHead(d_model=self.d_model, num_classes=self.num_classes, pool_type=self.pool_type)

    def reset_memory(self) -> None:
        if hasattr(self.nmm, "reset_memory"):
            self.nmm.reset_memory()
        if hasattr(self.temporal_retrieval, "reset_memory"):
            self.temporal_retrieval.reset_memory()

    def forward(
        self,
        x: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
        retrieval_x: Optional[torch.Tensor] = None,
        retrieval_mask: Optional[torch.Tensor] = None,
        retrieval_lags: Optional[torch.Tensor] = None,
        return_features: bool = False,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 先走主 route tokenization，得到 backbone 的基础输入表示。
        embeds, mask, main_route_aux = main_embedding_forward(self, x, sensor_mask)

        # temporal-detail route 输出统一整理为显式 time_state，便于后续分模块消费。
        temporal_route_output = None
        if self.temporal_detail_route is not None:
            if self.temporal_bridge_attn is not None:
                temporal_route_output = self.temporal_detail_route(
                    x,
                    sensor_mask=mask,
                    return_token_bundle=True,
                )
            else:
                temporal_route_output = self.temporal_detail_route(x, sensor_mask=mask)

        time_state: Dict[str, Optional[torch.Tensor]] = build_time_state_from_route_output(
            route_output=temporal_route_output,
            default_summary=torch.zeros_like(embeds),
        )
        temporal_detail_features = time_state["summary"]
        temporal_detail_tokens = select_temporal_bridge_tokens(self, time_state)

        temporal_detail_features = apply_temporal_bridge(
            self,
            base_features=embeds,
            temporal_tokens=temporal_detail_tokens,
            temporal_features=temporal_detail_features,
        )
        time_state["summary"] = temporal_detail_features

        primary_features = embeds
        temporal_fused_pre_backbone = False
        if self.use_temporal_pre_backbone_fusion:
            if self.temporal_fusion_stage == "adaptive_gate":
                # 用主路与时间路统计共同生成动态融合权重。
                gate_input = torch.cat([embeds, temporal_detail_features], dim=-1)
                weights = self.adaptive_gate(gate_input)
                w_time = weights[..., 0:1]
                w_sensor = weights[..., 1:2]
                primary_features = w_time * temporal_detail_features + w_sensor * embeds

            elif self.temporal_fusion_stage == "cross_attn_asym":
                # 用主路 query 从时间 token 中拉取局部细节。
                if temporal_detail_tokens is not None and self.cross_attn_asym is not None:
                    batch_size, num_vars, hidden_dim = embeds.shape
                    query = self.cross_attn_asym_q_norm(embeds.reshape(batch_size * num_vars, 1, hidden_dim))
                    kv = self.cross_attn_asym_kv_norm(temporal_detail_tokens)

                    attn_out, _ = self.cross_attn_asym(query, kv, kv, need_weights=False)
                    attn_out = attn_out.reshape(batch_size, num_vars, hidden_dim)

                    primary_features = embeds + self.cross_attn_asym_out_norm(attn_out)
                else:
                    primary_features = embeds + self.temporal_primary_residual_scale * temporal_detail_features

            elif self.temporal_route_role == "time_primary":
                primary_features = temporal_detail_features + (
                    self.temporal_primary_residual_scale * embeds
                )
            else:
                primary_features = primary_features + (
                    self.temporal_primary_residual_scale * temporal_detail_features
                )
            temporal_fused_pre_backbone = True

        # retrieval route 优先读取 time_state（segment/local/summary），兼容旧签名模块。
        retrieval_bundle = run_temporal_retrieval(
            retrieval_module=self.temporal_retrieval,
            x=x,
            route_features=primary_features,
            time_state=time_state,
            retrieval_x=retrieval_x,
            sensor_mask=mask,
            retrieval_mask=retrieval_mask,
            retrieval_lags=retrieval_lags,
        )

        backbone_input = primary_features

        # backbone 输出跨传感器表征，再由 NMM 追加记忆增强与 surprise 分数。
        sensor_state = self.backbone(backbone_input, mask)
        sensor_state, surprise = self.nmm(sensor_state)

        if return_features:
            feature_bundle = {
                # 新接口：显式 time/sensor 状态。
                "time_state": time_state,
                "sensor_state": sensor_state,
                # 兼容旧接口：保留原键名，避免下游立即断裂。
                "cross_sensor_features": sensor_state,
                "temporal_detail_features": temporal_detail_features,
                "temporal_detail_tokens": temporal_detail_tokens,
                "primary_route_features": primary_features,
                "surprise_scores": surprise,
                "temporal_route_role": self.temporal_route_role,
                "temporal_fusion_stage": self.temporal_fusion_stage,
                "temporal_bridge_mode": self.temporal_bridge_mode,
                "temporal_bridge_token_source": self.temporal_bridge_token_source,
                "temporal_fused_pre_backbone": temporal_fused_pre_backbone,
            }
            if main_route_aux is not None:
                feature_bundle["main_route_aux"] = main_route_aux
                component_bundle = main_route_aux.get("component")
                if isinstance(component_bundle, dict):
                    feature_bundle["component"] = component_bundle
            if retrieval_bundle is not None:
                feature_bundle["retrieval_bundle"] = retrieval_bundle
            return feature_bundle

        logits = self.head(sensor_state)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {
            "loss": loss,
            "labels": labels,
            "logits": logits,
            "surprise": surprise,
        }

