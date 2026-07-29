"""主干网络工厂函数。

该模块用于根据配置选择具体的 backbone 实现，
目前支持 Qwen 主干以及 Mamba / Mamba2 / Mamba3 混合主干。
"""

from copy import copy
from types import SimpleNamespace
from typing import Any

from ..layers.backbones import HybridBackbone, QwenBackbone


def _clone_model_cfg_with_mamba_version(model_cfg: Any, mamba_version: str) -> Any:
    """复制模型配置并覆盖 SSM 版本，避免直接修改原配置对象。"""

    cloned_cfg = copy(model_cfg)
    ssm_cfg = getattr(model_cfg, "ssm_cfg", SimpleNamespace())
    cloned_ssm_cfg = SimpleNamespace(**vars(ssm_cfg))
    cloned_ssm_cfg.mamba_version = mamba_version
    cloned_cfg.ssm_cfg = cloned_ssm_cfg
    return cloned_cfg


def get_backbone(cfg: Any) -> Any:
    """根据配置返回对应的 backbone 模块实例。"""

    backbone_type = str(getattr(cfg.model, "backbone_type", "mamba")).lower()

    if backbone_type == "qwen":
        return QwenBackbone(cfg.model)
    if backbone_type in {"mamba", "hybrid_mamba"}:
        return HybridBackbone(cfg.model)
    if backbone_type in {"mamba1", "mamba2", "mamba3"}:
        mamba_version = "mamba" if backbone_type == "mamba1" else backbone_type
        model_cfg = _clone_model_cfg_with_mamba_version(cfg.model, mamba_version)
        return HybridBackbone(model_cfg)

    # 未识别配置时，保守回退到当前默认主干，避免直接中断调试流程。
    print(f"Warning: backbone_type '{backbone_type}' not recognized, falling back to HybridBackbone (Mamba2)")
    return HybridBackbone(cfg.model)
