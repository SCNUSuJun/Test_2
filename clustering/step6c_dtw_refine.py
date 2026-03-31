"""
步骤6C：使用改进的DTW算法对初始类别进行轨迹相似度细化
=====================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from config import ClusterConfig
from utils import setup_logger


class ImprovedDTW:
    """改进 DTW（交叉步）。"""

    @staticmethod
    def compute_distance_matrix_single_pair(
        traj_a_lon: np.ndarray,
        traj_a_lat: np.ndarray,
        traj_b_lon: np.ndarray,
        traj_b_lat: np.ndarray,
    ) -> float:
        M, N = len(traj_a_lon), len(traj_b_lon)
        if M == 0 or N == 0:
            return float("inf")
        D = np.full((M + 1, N + 1), np.inf, dtype=np.float64)
        D[0, 0] = 0.0
        for m in range(1, M + 1):
            for n in range(1, N + 1):
                d = np.sqrt(
                    (traj_a_lon[m - 1] - traj_b_lon[n - 1]) ** 2
                    + (traj_a_lat[m - 1] - traj_b_lat[n - 1]) ** 2
                )
                c1 = D[m - 1, n - 1]
                c2 = D[m - 2, n - 1] if m >= 2 else np.inf
                c3 = D[m - 1, n - 2] if n >= 2 else np.inf
                D[m, n] = d + min(c1, c2, c3)
        return float(D[M, N])

    @staticmethod
    def compute_pairwise_dtw_matrix(
        trajectories_lon: List[np.ndarray],
        trajectories_lat: List[np.ndarray],
    ) -> np.ndarray:
        n = len(trajectories_lon)
        mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            mat[i, i] = 0.0
            for j in range(i + 1, n):
                d = ImprovedDTW.compute_distance_matrix_single_pair(
                    trajectories_lon[i],
                    trajectories_lat[i],
                    trajectories_lon[j],
                    trajectories_lat[j],
                )
                mat[i, j] = d
                mat[j, i] = d
        return mat


class TrajectoryRefiner:
    """轨迹相似度细化器。"""

    def __init__(self, config: ClusterConfig):
        self.cfg = config
        self.logger = setup_logger("TrajectoryRefiner")
        self.dtw = ImprovedDTW()

    def find_representative_trajectory(self, dtw_matrix: np.ndarray) -> int:
        n = dtw_matrix.shape[0]
        totals = np.zeros(n)
        for k in range(n):
            row = dtw_matrix[k].copy()
            row[k] = 0.0
            totals[k] = np.sum(row)
        return int(np.argmin(totals))

    def compute_split_threshold(self, dtw_matrix: np.ndarray, rep_idx: int) -> float:
        n = dtw_matrix.shape[0]
        dists = [dtw_matrix[rep_idx, j] for j in range(n) if j != rep_idx]
        if not dists:
            return float("inf")
        mu = float(np.mean(dists))
        sigma = float(np.std(dists))
        return mu + self.cfg.split_sigma_factor * sigma

    def _lon_lat_arrays(
        self, traj_indices: List[int], trajectories: List[pd.DataFrame]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        lons, lats = [], []
        for tid in traj_indices:
            t = trajectories[tid].sort_values("Timestamp", kind="mergesort")
            lons.append(t["LON"].to_numpy(dtype=np.float64))
            lats.append(t["LAT"].to_numpy(dtype=np.float64))
        return lons, lats

    def _cluster_dict(
        self,
        traj_indices: List[int],
        rep_local: int,
        trajectories: List[pd.DataFrame],
    ) -> Dict:
        rep_tid = traj_indices[rep_local]
        rt = trajectories[rep_tid].sort_values("Timestamp", kind="mergesort")
        return {
            "traj_indices": list(traj_indices),
            "rep_traj_idx": int(rep_tid),
            "rep_traj_lon": rt["LON"].to_numpy(dtype=np.float64),
            "rep_traj_lat": rt["LAT"].to_numpy(dtype=np.float64),
            "intra_avg_dist": 0.0,
        }

    def refine_single_group(
        self,
        traj_indices: List[int],
        trajectories: List[pd.DataFrame],
        depth: int = 0,
    ) -> List[Dict]:
        cfg = self.cfg
        if len(traj_indices) <= 1 or depth > cfg.max_refine_depth:
            return [self._cluster_dict(traj_indices, 0, trajectories)]
        if len(traj_indices) < cfg.min_cluster_size:
            return [self._cluster_dict(traj_indices, 0, trajectories)]

        lons, lats = self._lon_lat_arrays(traj_indices, trajectories)
        dtw_m = self.dtw.compute_pairwise_dtw_matrix(lons, lats)
        rep_local = self.find_representative_trajectory(dtw_m)
        thr = self.compute_split_threshold(dtw_m, rep_local)
        outlier_local = [
            j
            for j in range(len(traj_indices))
            if j != rep_local and dtw_m[rep_local, j] > thr
        ]
        if not outlier_local:
            d = self._cluster_dict(traj_indices, rep_local, trajectories)
            others = [
                dtw_m[rep_local, j]
                for j in range(len(traj_indices))
                if j != rep_local
            ]
            d["intra_avg_dist"] = float(np.mean(others)) if others else 0.0
            return [d]

        inlier_set = [traj_indices[j] for j in range(len(traj_indices)) if j not in outlier_local]
        outlier_set = [traj_indices[j] for j in outlier_local]
        res = []
        res.extend(self.refine_single_group(inlier_set, trajectories, depth + 1))
        res.extend(self.refine_single_group(outlier_set, trajectories, depth + 1))
        return res

    def run(
        self,
        initial_groups: Dict[Tuple[int, int], List[int]],
        trajectories: List[pd.DataFrame],
    ) -> List[Dict]:
        out: List[Dict] = []
        for key, idxs in initial_groups.items():
            if not idxs:
                continue
            if key == (-1, -1):
                for tid in idxs:
                    out.append(self._cluster_dict([tid], 0, trajectories))
                continue
            out.extend(self.refine_single_group(idxs, trajectories, 0))
        return out
