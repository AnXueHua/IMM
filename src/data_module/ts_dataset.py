"""时序数据集实现。

该模块负责数据读取、时间顺序切分、标准化、滑窗采样以及可选的历史检索候选构造，是整个实验的数据入口。"""

import os
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from . import register_dataset

warnings.filterwarnings("ignore")


def _get_ett_split_lengths(dataset: str, data_path: str, percent: float) -> tuple[int, int, int]:
    """Return the standard ETT train, test, and validation lengths."""

    dataset_identity = f"{dataset}/{os.path.basename(data_path)}".lower()
    sampling_multiplier = 4 if "ettm" in dataset_identity else 1
    total_train = 12 * 30 * 24 * sampling_multiplier
    num_train = int(total_train * (percent / 0.7))
    num_test = 4 * 30 * 24 * sampling_multiplier
    num_vali = 4 * 30 * 24 * sampling_multiplier
    return num_train, num_test, num_vali


# 该数据集类负责统一多变量时序预测任务中的数据读取、切分和滑窗样本构造。
@register_dataset("TSDataset")
class TSDataset(Dataset):
    """通用多变量时序预测数据集，支持标准预测和可选 retrieval 候选构造。"""

    def __init__(
        self,
        root_path: str,
        dataset: str,
        data_path: str,
        flag: str = "train",
        size: Tuple[int, int] = (512, 96),
        features: str = "M",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "h",
        percent: float = 1.0,
        use_retrieval: bool = False,
        retrieval_lags: Optional[Tuple[int, ...]] = None,
        retrieval_candidate_mode: str = "historical_context",
    ) -> None:
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.dataset = dataset
        self.data_path = data_path
        self.flag = flag
        self.use_retrieval = use_retrieval
        raw_lags = retrieval_lags or ()
        self.retrieval_lags = tuple(sorted({int(lag) for lag in raw_lags if int(lag) > 0}))
        self.retrieval_candidate_mode = str(retrieval_candidate_mode).lower()
        if self.retrieval_candidate_mode not in {"historical_context", "forecast_anchor"}:
            raise ValueError(
                "Unsupported retrieval_candidate_mode="
                f"{self.retrieval_candidate_mode}. Expected one of "
                "['historical_context', 'forecast_anchor']."
            )
        self.__read_data__()

    def __read_data__(self) -> None:
        # 先读取完整 CSV，再基于数据集类型决定 train/val/test 的时间切分方式。
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.dataset, self.data_path))

        num_train = int(len(df_raw) * self.percent)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        if "ett" in f"{self.dataset}/{self.data_path}".lower():
            num_train, num_test, num_vali = _get_ett_split_lengths(
                self.dataset,
                self.data_path,
                self.percent,
            )
        elif "illness" in self.dataset:
            num_train = int(len(df_raw) * self.percent)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test

        num_train = max(num_train, self.seq_len * 2)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[self.flag]
        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]

        if self.features in {"M", "MS"}:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            # 标准化只在训练段上拟合，避免验证/测试信息泄漏。
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            full_data = self.scaler.transform(df_data.values)
        else:
            full_data = df_data.values

        self.full_data = full_data
        self.data_x = full_data[self.border1:self.border2]
        self.data_y = full_data[self.border1:self.border2]
        self.num_vars = self.full_data.shape[1]

    def _build_retrieval_candidates(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 按预设 lag 从历史窗口中构造 retrieval 候选，供后续 retrieval/memory 分支使用。
        num_candidates = len(self.retrieval_lags)
        retrieval_x = torch.zeros(num_candidates, self.seq_len, self.num_vars, dtype=torch.float32)
        retrieval_mask = torch.zeros(num_candidates, dtype=torch.bool)
        retrieval_lags = torch.tensor(self.retrieval_lags, dtype=torch.float32)

        if num_candidates == 0:
            return retrieval_x, retrieval_mask, retrieval_lags

        global_start = self.border1 + index
        current_pred_start = global_start + self.seq_len
        for candidate_idx, lag in enumerate(self.retrieval_lags):
            if self.retrieval_candidate_mode == "forecast_anchor":
                retrieval_start = global_start - lag
                retrieval_end = retrieval_start + self.seq_len
                retrieval_future_end = retrieval_end + self.pred_len
                if retrieval_start < 0 or retrieval_future_end > current_pred_start:
                    continue
                retrieval_slice = self.full_data[retrieval_start:retrieval_end]
                if retrieval_slice.shape[0] != self.seq_len:
                    continue
                retrieval_x[candidate_idx] = torch.tensor(retrieval_slice, dtype=torch.float32)
                retrieval_mask[candidate_idx] = True
                continue

            retrieval_end = global_start - lag
            retrieval_start = retrieval_end - self.seq_len
            if retrieval_start < 0 or retrieval_end > global_start:
                continue
            retrieval_slice = self.full_data[retrieval_start:retrieval_end]
            if retrieval_slice.shape[0] != self.seq_len:
                continue
            retrieval_x[candidate_idx] = torch.tensor(retrieval_slice, dtype=torch.float32)
            retrieval_mask[candidate_idx] = True

        return retrieval_x, retrieval_mask, retrieval_lags

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # 每个样本由一个历史观测窗口 x 和一个未来预测窗口 y 组成。
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        sample = {
            "x": torch.tensor(seq_x, dtype=torch.float32),
            "y": torch.tensor(seq_y, dtype=torch.float32),
        }

        if self.use_retrieval and len(self.retrieval_lags) > 0:
            retrieval_x, retrieval_mask, retrieval_lags = self._build_retrieval_candidates(index)
            sample["retrieval_x"] = retrieval_x
            sample["retrieval_mask"] = retrieval_mask
            sample["retrieval_lags"] = retrieval_lags

        return sample

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)



