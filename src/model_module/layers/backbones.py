"""时序 backbone 实现集合。

该模块提供三类主干能力：
1. `MaskedAttentionLayer`：在变量维做带掩码的自注意力。
2. `MaskedMambaLayer`：在变量维做 Mamba 扫描，并支持双向扫描。
3. `HybridBackbone` / `QwenBackbone`：组装成完整的主干网络。
"""

import inspect
import os
from typing import Any

import torch
import torch.nn as nn


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def resolve_project_path(path: str) -> str:
    """把项目内相对模型路径解析为绝对路径。"""

    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


class MaskedAttentionLayer(nn.Module):
    """支持掩码的 Self-Attention 层，用于变量间关系建模。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # 前馈网络用于在注意力后进一步混合变量级特征。
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """在变量维执行带缺失掩码的注意力计算。"""

        # `key_padding_mask=True` 表示该变量位置在注意力中被忽略。
        key_padding_mask = ~(mask.bool())

        residual = x
        x_norm = self.norm(x)
        attn_out, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = residual + self.dropout(attn_out)
        x = x + self.ffn(x)
        return x


SUPPORTED_MAMBA_VERSIONS = {"mamba", "mamba1", "mamba2", "mamba3"}


def _get_mamba_class(mamba_version: str):
    """按配置选择 mamba_ssm 中的具体实现类。"""

    normalized = mamba_version.lower()
    if normalized not in SUPPORTED_MAMBA_VERSIONS:
        raise ValueError(
            f"Unsupported mamba_version={mamba_version}. "
            f"Expected one of {sorted(SUPPORTED_MAMBA_VERSIONS)}"
        )

    class_name = {
        "mamba": "Mamba",
        "mamba1": "Mamba",
        "mamba2": "Mamba2",
        "mamba3": "Mamba3",
    }[normalized]
    try:
        import mamba_ssm

        mamba_cls = getattr(mamba_ssm, class_name, None)
        if mamba_cls is not None:
            return mamba_cls
    except ImportError:
        pass

    module_path = {
        "Mamba": "mamba_ssm.modules.mamba_simple",
        "Mamba2": "mamba_ssm.modules.mamba2",
        "Mamba3": "mamba_ssm.modules.mamba3",
    }[class_name]
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"mamba_ssm does not expose {class_name}. "
            "Please verify the installed mamba_ssm version in the active environment."
        ) from exc


def _filter_init_kwargs(module_cls, candidate_kwargs: dict[str, Any]) -> dict[str, Any]:
    """只传递目标 Mamba 类显式支持的初始化参数。"""

    signature = inspect.signature(module_cls)
    supported = set(signature.parameters)
    return {
        key: value
        for key, value in candidate_kwargs.items()
        if key in supported and value is not None
    }


def _as_tuple(value: Any) -> Any:
    """把 YAML list 转成 Mamba 初始化更常用的 tuple。"""

    if isinstance(value, list):
        return tuple(value)
    return value


def _resolve_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    """把配置中的 dtype 名称解析为 torch dtype；为空时跟随全局训练精度。"""

    if not dtype_name:
        return None
    normalized = str(dtype_name).lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in dtype_map:
        raise ValueError(
            f"Unsupported ssm.dtype={dtype_name}. "
            f"Expected one of {sorted(dtype_map)} or null."
        )
    return dtype_map[normalized]


def _mamba1_kwargs(d_model: int, ssm_cfg, layer_idx: int | None) -> dict[str, Any]:
    """Mamba v1 参数面，避免把高版本参数误传给旧实现。"""

    return {
        "d_model": d_model,
        "d_state": getattr(ssm_cfg, "d_state", 16),
        "d_conv": getattr(ssm_cfg, "d_conv", 4),
        "expand": getattr(ssm_cfg, "expand", 2),
        "dt_rank": getattr(ssm_cfg, "mamba_dt_rank", "auto"),
        "dt_min": getattr(ssm_cfg, "mamba_dt_min", 0.001),
        "dt_max": getattr(ssm_cfg, "mamba_dt_max", 0.1),
        "dt_init": getattr(ssm_cfg, "mamba_dt_init", "random"),
        "dt_scale": getattr(ssm_cfg, "mamba_dt_scale", 1.0),
        "dt_init_floor": getattr(ssm_cfg, "mamba_dt_init_floor", 0.0001),
        "conv_bias": getattr(ssm_cfg, "mamba_conv_bias", True),
        "bias": getattr(ssm_cfg, "mamba_bias", False),
        "use_fast_path": getattr(ssm_cfg, "mamba_use_fast_path", True),
        "layer_idx": layer_idx,
        "dtype": _resolve_torch_dtype(getattr(ssm_cfg, "dtype", None)),
    }


def _mamba2_kwargs(d_model: int, ssm_cfg, layer_idx: int | None) -> dict[str, Any]:
    """Mamba2 参数面，默认值贴近 mamba_ssm 官方签名。"""

    return {
        "d_model": d_model,
        "d_state": getattr(ssm_cfg, "d_state", 128),
        "d_conv": getattr(ssm_cfg, "d_conv", 4),
        "conv_init": getattr(ssm_cfg, "mamba2_conv_init", None),
        "expand": getattr(ssm_cfg, "expand", 2),
        "headdim": getattr(ssm_cfg, "headdim", 64),
        "d_ssm": getattr(ssm_cfg, "mamba2_d_ssm", None),
        "ngroups": getattr(ssm_cfg, "ngroups", 1),
        "A_init_range": _as_tuple(getattr(ssm_cfg, "mamba2_a_init_range", (1, 16))),
        "D_has_hdim": getattr(ssm_cfg, "mamba2_d_has_hdim", False),
        "rmsnorm": getattr(ssm_cfg, "mamba2_rmsnorm", True),
        "norm_before_gate": getattr(ssm_cfg, "mamba2_norm_before_gate", False),
        "dt_min": getattr(ssm_cfg, "mamba_dt_min", 0.001),
        "dt_max": getattr(ssm_cfg, "mamba_dt_max", 0.1),
        "dt_init_floor": getattr(ssm_cfg, "mamba_dt_init_floor", 0.0001),
        "dt_limit": _as_tuple(getattr(ssm_cfg, "mamba2_dt_limit", (0.0, float("inf")))),
        "bias": getattr(ssm_cfg, "mamba_bias", False),
        "conv_bias": getattr(ssm_cfg, "mamba_conv_bias", True),
        "chunk_size": getattr(ssm_cfg, "chunk_size", 256),
        "use_mem_eff_path": getattr(ssm_cfg, "mamba2_use_mem_eff_path", True),
        "sequence_parallel": getattr(ssm_cfg, "mamba2_sequence_parallel", True),
        "layer_idx": layer_idx,
        "dtype": _resolve_torch_dtype(getattr(ssm_cfg, "dtype", None)),
    }


def _mamba3_kwargs(
    d_model: int,
    ssm_cfg,
    layer_idx: int | None,
    n_layer: int | None,
) -> dict[str, Any]:
    """Mamba3 参数面，包含官方 MIMO 相关参数。"""

    return {
        "d_model": d_model,
        "d_state": getattr(ssm_cfg, "d_state", 128),
        "expand": getattr(ssm_cfg, "expand", 2),
        "headdim": getattr(ssm_cfg, "headdim", 64),
        "ngroups": getattr(ssm_cfg, "ngroups", 1),
        "rope_fraction": getattr(ssm_cfg, "mamba3_rope_fraction", 0.5),
        "dt_min": getattr(ssm_cfg, "mamba_dt_min", 0.001),
        "dt_max": getattr(ssm_cfg, "mamba_dt_max", 0.1),
        "dt_init_floor": getattr(ssm_cfg, "mamba_dt_init_floor", 0.0001),
        "A_floor": getattr(ssm_cfg, "mamba3_a_floor", 0.0001),
        "is_outproj_norm": getattr(ssm_cfg, "mamba3_is_outproj_norm", False),
        "is_mimo": getattr(ssm_cfg, "mamba3_is_mimo", False),
        "mimo_rank": getattr(ssm_cfg, "mamba3_mimo_rank", 4),
        "chunk_size": getattr(ssm_cfg, "chunk_size", 64),
        "dropout": getattr(ssm_cfg, "mamba3_dropout", 0.0),
        "layer_idx": layer_idx,
        "n_layer": n_layer,
        "dtype": _resolve_torch_dtype(getattr(ssm_cfg, "dtype", None)),
    }


def _build_mamba_block(
    d_model: int,
    ssm_cfg,
    layer_idx: int | None,
    n_layer: int | None,
) -> nn.Module:
    """构造指定版本的 Mamba block，并保持旧 Mamba2 参数兼容。"""

    mamba_version = str(getattr(ssm_cfg, "mamba_version", "mamba2")).lower()
    mamba_cls = _get_mamba_class(mamba_version)
    if mamba_version in {"mamba", "mamba1"}:
        candidate_kwargs = _mamba1_kwargs(d_model, ssm_cfg, layer_idx)
    elif mamba_version == "mamba2":
        candidate_kwargs = _mamba2_kwargs(d_model, ssm_cfg, layer_idx)
    else:
        candidate_kwargs = _mamba3_kwargs(d_model, ssm_cfg, layer_idx, n_layer)
    return mamba_cls(**_filter_init_kwargs(mamba_cls, candidate_kwargs))


class MaskedMambaLayer(nn.Module):
    """支持变量掩码的 Mamba 层，可在 Mamba / Mamba2 / Mamba3 间切换。"""

    def __init__(self, d_model: int, ssm_cfg, layer_idx: int | None = None, n_layer: int | None = None):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.bidirectional = getattr(ssm_cfg, "bidirectional", True)
        self.mamba_version = str(getattr(ssm_cfg, "mamba_version", "mamba2")).lower()

        try:
            self.mamba_fwd = _build_mamba_block(
                d_model=d_model,
                ssm_cfg=ssm_cfg,
                layer_idx=layer_idx,
                n_layer=n_layer,
            )

            if self.bidirectional:
                self.mamba_bwd = _build_mamba_block(
                    d_model=d_model,
                    ssm_cfg=ssm_cfg,
                    layer_idx=layer_idx,
                    n_layer=n_layer,
                )
        except (ImportError, ValueError) as exc:
            allow_fallback = getattr(ssm_cfg, "allow_fallback", False)
            if not allow_fallback:
                raise ImportError(
                    f"mamba_ssm with {self.mamba_version} support is required for benchmark runs. "
                    "Set ssm_cfg.allow_fallback=True only for explicit debug usage."
                ) from exc
            print(f"Warning: mamba_ssm {self.mamba_version} unavailable, using Linear fallback")
            self.mamba_fwd = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model),
            )
            self.bidirectional = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """在变量维执行 Mamba 扫描，并屏蔽缺失传感器。"""

        residual = x
        x_norm = self.norm(x)

        # 在状态空间扫描前先对缺失变量清零，避免无效传感器污染状态。
        mask_expanded = mask.unsqueeze(-1).to(x.dtype)
        x_masked = x_norm * mask_expanded

        out_fwd = self.mamba_fwd(x_masked)

        if self.bidirectional:
            # 反向扫描用于减轻人为传感器排序带来的方向性偏置。
            x_flipped = torch.flip(x_masked, dims=[1])
            out_bwd = self.mamba_bwd(x_flipped)
            out_bwd = torch.flip(out_bwd, dims=[1])
            out = (out_fwd + out_bwd) / 2
        else:
            out = out_fwd

        out = out * mask_expanded
        return residual + out


class HybridBackbone(nn.Module):
    """Jamba 风格的混合主干，按层交替使用 Mamba 与 Attention。"""

    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_layer = cfg.n_layer
        self.attn_offset = getattr(cfg, "attn_layer_offset", 1)
        self.attn_period = getattr(cfg, "attn_layer_period", 4)

        self.layers = nn.ModuleList()
        for i in range(self.n_layer):
            is_attn = (i >= self.attn_offset) and ((i - self.attn_offset) % self.attn_period == 0)
            if is_attn:
                self.layers.append(
                    MaskedAttentionLayer(
                        self.d_model,
                        getattr(cfg, "num_heads", 8),
                        getattr(cfg, "dropout", 0.1),
                    )
                )
            else:
                self.layers.append(
                    MaskedMambaLayer(
                        self.d_model,
                        cfg.ssm_cfg,
                        layer_idx=i,
                        n_layer=self.n_layer,
                    )
                )

        self.norm_f = nn.LayerNorm(self.d_model)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """顺序通过所有混合层，并在末尾做统一归一化。"""

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)

        hidden_states = self.norm_f(hidden_states)
        return hidden_states


class QwenBackbone(nn.Module):
    """把变量 token 直接送入预训练 Qwen 主干进行关系建模。"""

    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModel
        from transformers.utils.logging import disable_progress_bar

        qwen_path = resolve_project_path(getattr(cfg, "qwen_path", "Qwen/Qwen3.5-0.8B"))
        self.d_model = cfg.d_model

        disable_progress_bar()
        print(f"Loading Qwen backbone from {qwen_path}...")
        self.qwen = AutoModel.from_pretrained(qwen_path, trust_remote_code=True)
        print("Qwen backbone loaded successfully.")

        freeze_qwen = getattr(cfg, "freeze_qwen", True)
        if freeze_qwen:
            for param in self.qwen.parameters():
                param.requires_grad = False

        self.qwen_hidden_size = self.qwen.config.hidden_size
        self.in_proj = nn.Linear(self.d_model, self.qwen_hidden_size) if self.d_model != self.qwen_hidden_size else nn.Identity()
        self.out_proj = nn.Linear(self.qwen_hidden_size, self.d_model) if self.d_model != self.qwen_hidden_size else nn.Identity()

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """把变量级隐状态送入 Qwen，并投影回时序主干维度。"""

        x = self.in_proj(hidden_states)
        outputs = self.qwen(
            inputs_embeds=x,
            attention_mask=mask,
            output_hidden_states=False,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state
        out = self.out_proj(last_hidden_state)

        # 再次清理掉缺失位置，避免冻结 backbone 输出激活无效变量。
        mask_expanded = mask.unsqueeze(-1).to(out.dtype)
        out = out * mask_expanded

        return out
