"""
方案 11.4 依赖的可运行默认实现（内存统计 + 空几何占位）。

真实航道几何需替换为 ChannelGeometryStore / ForkJunctionRegistry 的具体实现。
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Set, Tuple

from resources.fork_resources import (
    ChannelGeometryStore,
    ForkJunctionRegistry,
    MmsiBranchStatsStore,
)
from utils import load_json


class NullChannelGeometryStore(ChannelGeometryStore):
    """无几何数据：距离为 NaN，邻近航道列表为空。方案 11.4 第 4 步需真实 Store。"""

    def distance_to_centerline_m(
        self, lon: float, lat: float, channel_id: str
    ) -> float:
        return float("nan")

    def list_channel_ids_near(
        self, lon: float, lat: float, radius_m: float
    ) -> List[str]:
        return []


class NullForkJunctionRegistry(ForkJunctionRegistry):
    """未注册分叉口。"""

    def clusters_at_junction(self, junction_id: str) -> Set[int]:
        return set()

    def junction_for_location(self, lon: float, lat: float) -> Optional[str]:
        return None


class InMemoryMmsiBranchStatsStore(MmsiBranchStatsStore):
    """MMSI × cluster_id 计数，用于 P(C_k) = N_k / sum N_j。"""

    def __init__(self) -> None:
        self._counts: Dict[Tuple[int, int], int] = {}

    def counts_for_mmsi_clusters(
        self, mmsi: int, cluster_ids: List[int]
    ) -> Dict[int, int]:
        return {c: self._counts.get((mmsi, c), 0) for c in cluster_ids}

    def record_trajectory_assignment(
        self, mmsi: int, cluster_id: int, traj_id: str
    ) -> None:
        key = (mmsi, cluster_id)
        self._counts[key] = self._counts.get(key, 0) + 1

    def seed_from_counts(self, mmsi: int, cluster_counts: Dict[int, int]) -> None:
        """批量设置某 MMSI 在各 cluster 上的历史条数。"""
        for cid, n in cluster_counts.items():
            self._counts[(mmsi, int(cid))] = int(n)

    def load_from_offline_json(self, path: str) -> int:
        """
        加载离线流水线写入的 mmsi_cluster_counts.json（方案 11.4 N_k）。

        Returns
        -------
        n_loaded : int
            写入的 (MMSI, cluster) 键数量。
        """
        if not path or not os.path.isfile(path):
            return 0
        try:
            data = load_json(path)
        except (OSError, ValueError, TypeError):
            return 0
        mc = data.get("mmsi_clusters") or {}
        n = 0
        for sm, inner in mc.items():
            try:
                m = int(sm)
            except (TypeError, ValueError):
                continue
            if not isinstance(inner, dict):
                continue
            for sc, cnt in inner.items():
                try:
                    cid = int(sc)
                    self._counts[(m, cid)] = int(cnt)
                    n += 1
                except (TypeError, ValueError):
                    continue
        return n


def prune_branch_predictions_by_geometry(
    geometry_store: Optional[ChannelGeometryStore],
    obs_lon: float,
    obs_lat: float,
    branch_preds: Dict[int, List[Dict]],
    deviation_m: float,
    search_radius_m: float = 800.0,
) -> Dict[int, List[Dict]]:
    """
    方案 11.4 第 4 步的工程钩子：用航道中心线距离粗筛分支预测。
    NullChannelGeometryStore 或未找到航道时原样返回。
    """
    if not branch_preds:
        return branch_preds
    if geometry_store is None or isinstance(
        geometry_store, NullChannelGeometryStore
    ):
        return branch_preds
    try:
        ch_ids = geometry_store.list_channel_ids_near(
            obs_lon, obs_lat, search_radius_m
        )
    except Exception:
        return branch_preds
    if not ch_ids:
        return branch_preds
    kept: Dict[int, List[Dict]] = {}
    for cid, pts in branch_preds.items():
        if not pts:
            continue
        p0 = pts[0]
        try:
            lon = float(p0.get("lon", p0.get("LON", float("nan"))))
            lat = float(p0.get("lat", p0.get("LAT", float("nan"))))
        except (TypeError, ValueError):
            kept[cid] = pts
            continue
        dmin = float("inf")
        for ch in ch_ids:
            try:
                d = geometry_store.distance_to_centerline_m(lon, lat, ch)
            except Exception:
                continue
            if math.isfinite(d):
                dmin = min(dmin, d)
        if not math.isfinite(dmin):
            kept[cid] = pts
        elif dmin <= deviation_m:
            kept[cid] = pts
    return kept if kept else branch_preds
