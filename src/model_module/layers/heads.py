"""任务输出头。

该模块把变量级特征聚合成样本级表示，再映射到最终分类空间。
当前主要用于分类类任务，不直接参与预测解码。
"""

import torch
import torch.nn as nn


class OutputHead(nn.Module):
    """将 `[B, V, D]` 特征压缩为分类 logits。"""

    def __init__(self, d_model: int, num_classes: int, pool_type: str = "mean"):
        super().__init__()
        self.pool_type = pool_type

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """先在变量维聚合，再输出分类结果。"""

        if self.pool_type == "mean":
            pooled = x.mean(dim=1)
        elif self.pool_type == "max":
            pooled = x.max(dim=1).values
        else:
            # 未识别配置时回退到均值池化，保持输出稳定。
            pooled = x.mean(dim=1)

        logits = self.classifier(pooled)
        return logits
