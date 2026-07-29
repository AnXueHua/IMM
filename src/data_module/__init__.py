"""数据集工厂与自动注册入口。

该模块负责：
1. 提供 `register_dataset` 装饰器，统一注册数据集类。
2. 提供 `DatasetFactory`，按名称实例化数据集对象。
3. 自动导入当前目录下的子模块，触发各数据集文件中的注册逻辑。
"""

import importlib
import os
from typing import Dict, Type

from torch.utils.data import Dataset

DATASET_FACTORY: Dict[str, Type[Dataset]] = {}


def register_dataset(name: str):
    """注册数据集类到全局工厂表。"""

    def decorator(cls):
        DATASET_FACTORY[name] = cls
        return cls

    return decorator


def DatasetFactory(name: str, *args, **kwargs) -> Dataset:
    """根据名称构造数据集实例。"""

    dataset_cls = DATASET_FACTORY.get(name)
    if dataset_cls is None:
        raise ValueError(f"Dataset {name} is not registered in DATASET_FACTORY.")
    return dataset_cls(*args, **kwargs)


# 自动导入当前目录下的 Python 文件与子目录，确保注册装饰器在导入时生效。
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith("_") and not file.startswith(".") and (file.endswith(".py") or os.path.isdir(path)):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module(f"src.data_module.{model_name}")

__all__ = ["DATASET_FACTORY", "register_dataset", "DatasetFactory"]
