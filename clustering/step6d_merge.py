"""
步骤6D：跨类别合并——处理方向相反但路线相同的轨迹
==================================================
方案 6.4：SimDist = min(DTW, DTW/rev)；θ_merge 合并；合并后方向统一 + 特征轨迹重算。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set

from config import ClusterConfig
from clustering.step6c_dtw_refine import ImprovedDTW
from utils import setup_logger


class ClusterMerger:
    """跨类别合并器。"""

    def __init__(self, config: ClusterConfig):
        self.cfg = config
        self.logger = setup_logger("ClusterMerger")
        self.dtw = ImprovedDTW()

    def compute_inter_cluster_distance(
        self,
        feat_traj_i_lon: np.ndarray,
        feat_traj_i_lat: np.ndarray,
        feat_traj_j_lon: np.ndarray,
        feat_traj_j_lat: np.ndarray,
    ) -> float:
        d1 = self.dtw.compute_distance_matrix_single_pair(
            feat_traj_i_lon,
            feat_traj_i_lat,
            feat_traj_j_lon,
            feat_traj_j_lat,
        )
        d2 = self.dtw.compute_distance_matrix_single_pair(
            feat_traj_i_lon,
            feat_traj_i_lat,
            feat_traj_j_lon[::-1],
            feat_traj_j_lat[::-1],
        )
        return float(min(d1, d2))

    def _dtw_symmetric_traj_pair(
        self,
        lon_a: np.ndarray,
        lat_a: np.ndarray,
        lon_b: np.ndarray,
        lat_b: np.ndarray,
    ) -> float:
        """类内/类间对称：min(DTW(A,B), DTW(A,reverse(B)))。"""
        if len(lon_a) == 0 or len(lon_b) == 0:
            return float("inf")
        d1 = self.dtw.compute_distance_matrix_single_pair(
            lon_a, lat_a, lon_b, lat_b
        )
        d2 = self.dtw.compute_distance_matrix_single_pair(
            lon_a, lat_a, lon_b[::-1], lat_b[::-1]
        )
        return float(min(d1, d2))

    def find_merge_pairs(
        self, clusters: List[Dict], trajectories: List[pd.DataFrame]
    ) -> List[Tuple[int, int]]:
        n = len(clusters)
        if n < 2:
            return []
        dists: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                Fi = clusters[i]
                Fj = clusters[j]
                d = self.compute_inter_cluster_distance(
                    Fi["rep_traj_lon"],
                    Fi["rep_traj_lat"],
                    Fj["rep_traj_lon"],
                    Fj["rep_traj_lat"],
                )
                dists.append(d)
        if self.cfg.merge_threshold is not None:
            thr = float(self.cfg.merge_threshold)
        else:
            if not dists:
                return []
            mu = float(np.mean(dists))
            sigma = float(np.std(dists))
            thr = mu + float(self.cfg.merge_auto_sigma_factor) * sigma
            self.logger.info(
                "6D θ_merge 自动：mean(SimDist)=%.6g σ=%.6g thr=μ+kσ=%.6g (k=%s)",
                mu,
                sigma,
                thr,
                self.cfg.merge_auto_sigma_factor,
            )
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                Fi, Fj = clusters[i], clusters[j]
                d = self.compute_inter_cluster_distance(
                    Fi["rep_traj_lon"],
                    Fi["rep_traj_lat"],
                    Fj["rep_traj_lon"],
                    Fj["rep_traj_lat"],
                )
                if d < thr:
                    pairs.append((i, j))
        return pairs

    def _uf_find(self, parent: List[int], x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _uf_union(self, parent: List[int], i: int, j: int) -> None:
        ri, rj = self._uf_find(parent, i), self._uf_find(parent, j)
        if ri != rj:
            parent[rj] = ri

    def _lon_lat_forward(
        self, tid: int, trajectories: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        g = trajectories[tid].sort_values("Timestamp", kind="mergesort")
        return (
            g["LON"].to_numpy(dtype=np.float64),
            g["LAT"].to_numpy(dtype=np.float64),
        )

    def finalize_cluster_representative(
        self, cluster: Dict, trajectories: List[pd.DataFrame]
    ) -> Dict:
        """
        方案 6.4.2 第 3 步：方向统一 + 6E 特征轨迹 = 与类内其他轨迹改进 DTW（对称）距离和最小者。
        输出 orientation_flip: traj_id -> 是否在步骤 7/8 前做时间逆序。
        """
        idxs: List[int] = sorted(cluster["traj_indices"])
        if not idxs:
            cluster["orientation_flip"] = {}
            return cluster
        if len(idxs) == 1:
            tid = idxs[0]
            lon, lat = self._lon_lat_forward(tid, trajectories)
            cluster["rep_traj_idx"] = int(tid)
            cluster["rep_traj_lon"] = lon.copy()
            cluster["rep_traj_lat"] = lat.copy()
            cluster["orientation_flip"] = {int(tid): False}
            return cluster

        best_tid: int = idxs[0]
        best_total = float("inf")
        lon_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
            t: self._lon_lat_forward(t, trajectories) for t in idxs
        }

        for c in idxs:
            lon_c, lat_c = lon_cache[c]
            total = 0.0
            for j in idxs:
                if j == c:
                    continue
                lon_j, lat_j = lon_cache[j]
                total += self._dtw_symmetric_traj_pair(lon_c, lat_c, lon_j, lat_j)
            if total < best_total:
                best_total = total
                best_tid = c

        rep_lon, rep_lat = lon_cache[best_tid]
        cluster["rep_traj_idx"] = int(best_tid)
        cluster["rep_traj_lon"] = np.asarray(rep_lon, dtype=np.float64).copy()
        cluster["rep_traj_lat"] = np.asarray(rep_lat, dtype=np.float64).copy()

        flips: Dict[int, bool] = {}
        for j in idxs:
            if j == best_tid:
                flips[int(j)] = False
                continue
            lon_j, lat_j = lon_cache[j]
            df = self.dtw.compute_distance_matrix_single_pair(
                cluster["rep_traj_lon"],
                cluster["rep_traj_lat"],
                lon_j,
                lat_j,
            )
            dr = self.dtw.compute_distance_matrix_single_pair(
                cluster["rep_traj_lon"],
                cluster["rep_traj_lat"],
                lon_j[::-1],
                lat_j[::-1],
            )
            flips[int(j)] = bool(dr < df)
        cluster["orientation_flip"] = flips
        return cluster

    def merge_clusters(
        self,
        clusters: List[Dict],
        merge_pairs: List[Tuple[int, int]],
    ) -> List[Dict]:
        n = len(clusters)
        if n == 0:
            return []
        parent = list(range(n))
        for i, j in merge_pairs:
            self._uf_union(parent, i, j)
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            r = self._uf_find(parent, i)
            groups.setdefault(r, []).append(i)
        merged: List[Dict] = []
        for _, members in groups.items():
            all_idx: Set[int] = set()
            for mi in sorted(members):
                all_idx.update(clusters[mi]["traj_indices"])
            merged.append(
                {
                    "traj_indices": sorted(all_idx),
                    "intra_avg_dist": 0.0,
                }
            )
        return merged

    def compute_cluster_quality(
        self, clusters: List[Dict], trajectories: List[pd.DataFrame]
    ) -> List[Dict]:
        for c in clusters:
            rep_tid = int(c["rep_traj_idx"])
            rl = c.get("rep_traj_lon")
            ra = c.get("rep_traj_lat")
            if rl is None or ra is None:
                lon, lat = self._lon_lat_forward(rep_tid, trajectories)
                rl = np.asarray(lon, dtype=np.float64)
                ra = np.asarray(lat, dtype=np.float64)
                c["rep_traj_lon"] = rl
                c["rep_traj_lat"] = ra
            dists = []
            for tid in c["traj_indices"]:
                if tid == rep_tid:
                    continue
                lon_j, lat_j = self._lon_lat_forward(tid, trajectories)
                d = self._dtw_symmetric_traj_pair(
                    np.asarray(rl, dtype=np.float64),
                    np.asarray(ra, dtype=np.float64),
                    lon_j,
                    lat_j,
                )
                dists.append(d)
            c["intra_avg_dist"] = float(np.mean(dists)) if dists else 0.0
        return clusters

    def assign_final_cluster_ids(self, clusters: List[Dict]) -> List[Dict]:
        for i, c in enumerate(clusters):
            c["cluster_id"] = i
        return clusters

    def run(
        self, refined_clusters: List[Dict], trajectories: List[pd.DataFrame]
    ) -> List[Dict]:
        pairs = self.find_merge_pairs(refined_clusters, trajectories)
        if pairs:
            merged = self.merge_clusters(refined_clusters, pairs)
        else:
            merged = []
            for c in refined_clusters:
                merged.append(
                    {
                        "traj_indices": list(c["traj_indices"]),
                        "intra_avg_dist": float(c.get("intra_avg_dist", 0.0)),
                    }
                )
        finalized = [
            self.finalize_cluster_representative(dict(cl), trajectories)
            for cl in merged
        ]
        finalized = self.compute_cluster_quality(finalized, trajectories)
        return self.assign_final_cluster_ids(finalized)
