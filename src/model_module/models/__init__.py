"""核心模型注册表与模型导出入口。

该目录下存放完整模型架构，区别于 `layers/` 中的基础网络层。
这里统一维护模型注册表，方便实验脚本按名称构造模型。
"""

from __future__ import annotations

from typing import Callable, Dict, Type

MODEL_REGISTRY: Dict[str, Type[object]] = {}


def register_model(name: str) -> Callable[[Type[object]], Type[object]]:
    """将模型类注册到全局模型表。"""

    def decorator(cls: Type[object]) -> Type[object]:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


from .imamba_memory import iMambaMemoryModel
from .forecast_core import ForecastCore
from .imm_llm import IMM_LLMModel

__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "ForecastCore",
    "iMambaMemoryModel",
    "IMM_LLMModel",
]
