"""
步骤8：为每个聚类类别构造LSTM训练样本
======================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader

from config import TrainConfig, SplitConfig
from utils import (
    compute_curvature_coefficient,
    get_T_from_curvature,
    get_H_from_curvature,
    setup_logger,
)


class TrajectoryDataset(torch.utils.data.Dataset):
    """PyTorch Dataset：LSTM (input, label) 样本对。"""

    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        self.inputs = torch.FloatTensor(inputs)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]


class SampleConstructor:
    """LSTM 训练样本构造器。"""

    def __init__(
        self,
        config: TrainConfig,
        split_config: Optional[SplitConfig] = None,
    ):
        self.cfg = config
        self.split_cfg = split_config
        self.logger = setup_logger("SampleConstructor")

    def _representative_year(self, traj: pd.DataFrame) -> int:
        """轨迹代表年：Timestamp.max() 的 UTC 日历年（naive Timestamp 语义为 UTC，见步骤1）。"""
        ts = traj["Timestamp"].max()
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return int(t.year)

    def _time_based_split_indices(
        self, normalized_trajs: List[pd.DataFrame]
    ) -> Tuple[List[int], List[int], List[int], int]:
        sc = self.split_cfg
        assert sc is not None
        train_set = set(sc.train_years)
        val_set = set(sc.val_years)
        test_set = set(sc.test_years)
        tr, va, te = [], [], []
        unknown = 0
        for i, df in enumerate(normalized_trajs):
            y = self._representative_year(df)
            if y in train_set:
                tr.append(i)
            elif y in val_set:
                va.append(i)
            elif y in test_set:
                te.append(i)
            else:
                unknown += 1
        return tr, va, te, unknown

    def determine_time_steps(
        self, cluster_info: Dict, trajectories: List[pd.DataFrame]
    ) -> int:
        rl = cluster_info.get("rep_traj_lon")
        ra = cluster_info.get("rep_traj_lat")
        if rl is not None and ra is not None:
            lons = np.asarray(rl, dtype=np.float64)
            lats = np.asarray(ra, dtype=np.float64)
        else:
            rep = int(cluster_info["rep_traj_idx"])
            t = trajectories[rep].sort_values("Timestamp", kind="mergesort")
            lons = t["LON"].to_numpy(dtype=np.float64)
            lats = t["LAT"].to_numpy(dtype=np.float64)
        C = compute_curvature_coefficient(lons, lats)
        return get_T_from_curvature(C, dict(self.cfg.curvature_to_T))

    def determine_hidden_size(
        self, cluster_info: Dict, trajectories: List[pd.DataFrame]
    ) -> int:
        rl = cluster_info.get("rep_traj_lon")
        ra = cluster_info.get("rep_traj_lat")
        if rl is not None and ra is not None:
            lons = np.asarray(rl, dtype=np.float64)
            lats = np.asarray(ra, dtype=np.float64)
        else:
            rep = int(cluster_info["rep_traj_idx"])
            t = trajectories[rep].sort_values("Timestamp", kind="mergesort")
            lons = t["LON"].to_numpy(dtype=np.float64)
            lats = t["LAT"].to_numpy(dtype=np.float64)
        C = compute_curvature_coefficient(lons, lats)
        return get_H_from_curvature(C, dict(self.cfg.curvature_to_H))

    def generate_samples_single_trajectory(
        self, normalized_traj: pd.DataFrame, T: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        g = normalized_traj.sort_values("Timestamp", kind="mergesort").reset_index(
            drop=True
        )
        feats = ["LON", "LAT", "SOG", "COG"]
        arr = g[feats].to_numpy(dtype=np.float64)
        m = len(arr)
        if m <= T:
            return np.zeros((0, T, 4)), np.zeros((0, 4))
        n_samples = m - T
        X = np.zeros((n_samples, T, 4), dtype=np.float64)
        y = np.zeros((n_samples, 4), dtype=np.float64)
        for j in range(n_samples):
            X[j] = arr[j : j + T]
            y[j] = arr[j + T]
        return X, y

    def split_by_trajectory(
        self, normalized_trajs: List[pd.DataFrame], T: int
    ) -> Tuple[TrajectoryDataset, TrajectoryDataset, TrajectoryDataset]:
        n = len(normalized_trajs)
        if (
            self.split_cfg is not None
            and self.split_cfg.time_based_split
            and self.split_cfg.split_by_voyage
        ):
            train_list, val_list, test_list, n_unknown = self._time_based_split_indices(
                normalized_trajs
            )
            if n_unknown:
                self.logger.warning(
                    "步骤8 时间划分：%s 条轨迹未落入 train/val/test 年份集合，已排除",
                    n_unknown,
                )
            train_idx = np.asarray(train_list, dtype=int)
            val_idx = np.asarray(val_list, dtype=int)
            test_idx = np.asarray(test_list, dtype=int)
        else:
            rng = np.random.RandomState(42)
            perm = rng.permutation(n)
            n_tr = int(round(n * self.cfg.train_ratio))
            n_va = int(round(n * self.cfg.val_ratio))
            train_idx = perm[:n_tr]
            val_idx = perm[n_tr : n_tr + n_va]
            test_idx = perm[n_tr + n_va :]

        def build_dataset(idxs: np.ndarray) -> TrajectoryDataset:
            Xs, Ys = [], []
            for i in idxs:
                X, y = self.generate_samples_single_trajectory(normalized_trajs[i], T)
                if len(X) == 0:
                    continue
                Xs.append(X)
                Ys.append(y)
            if not Xs:
                return TrajectoryDataset(
                    np.zeros((0, T, 4)), np.zeros((0, 4))
                )
            return TrajectoryDataset(np.concatenate(Xs), np.concatenate(Ys))

        return (
            build_dataset(train_idx),
            build_dataset(val_idx),
            build_dataset(test_idx),
        )

    def create_dataloaders(
        self,
        train_ds: TrajectoryDataset,
        val_ds: TrajectoryDataset,
        test_ds: TrajectoryDataset,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        bs = self.cfg.batch_size
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False),
            DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False),
            DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False),
        )

    def run(
        self,
        cluster_info: Dict,
        normalized_trajs: List[pd.DataFrame],
        trajectories: List[pd.DataFrame],
    ) -> Dict:
        T = self.determine_time_steps(cluster_info, trajectories)
        H = self.determine_hidden_size(cluster_info, trajectories)
        tr_ds, va_ds, te_ds = self.split_by_trajectory(normalized_trajs, T)
        tr_l, va_l, te_l = self.create_dataloaders(tr_ds, va_ds, te_ds)
        return {
            "T": T,
            "H": H,
            "train_loader": tr_l,
            "val_loader": va_l,
            "test_loader": te_l,
            "train_dataset": tr_ds,
            "val_dataset": va_ds,
            "test_dataset": te_ds,
        }
