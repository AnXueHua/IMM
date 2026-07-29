"""基础网络层导出入口。

`layers/` 只保留可复用的底层组件，例如 embedding、backbone、memory、
temporal route、retrieval 和输出头等，不直接承担完整模型编排。
"""

from .backbones import HybridBackbone, MaskedAttentionLayer, MaskedMambaLayer, QwenBackbone
from .embeddings import DecomposedInvertedEmbedding, InvertedEmbedding, MixedInvertedEmbedding
from .heads import OutputHead
from .nmm import NeuralMemoryMatrix
from .temporal_decomposition import TemporalDecomposition
from .temporal_detail import TemporalDetailRoute
from .temporal_retrieval import PeriodicRegimeRetriever, TemporalPatternRetriever

__all__ = [
    "InvertedEmbedding",
    "MixedInvertedEmbedding",
    "DecomposedInvertedEmbedding",
    "HybridBackbone",
    "QwenBackbone",
    "MaskedMambaLayer",
    "MaskedAttentionLayer",
    "NeuralMemoryMatrix",
    "OutputHead",
    "TemporalDetailRoute",
    "TemporalDecomposition",
    "TemporalPatternRetriever",
    "PeriodicRegimeRetriever",
]
