"""
方案 11.4 依赖的可运行默认实现（内存统计 + 空几何占位 + 可选 GeoJSON/JSON 加载）。

- GeoJsonLineStringChannelStore：中心线距离与邻近航道查询，供几何剪枝。
- JsonForkJunctionRegistry：分叉口圆域注册；若需「仅接近分叉口才进入多分支」，
  可在 step11 `detect_fork_situation` 中与 `junction_for_location` 组合（当前未强制门控）。
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from resources.fork_resources import (
    ChannelGeometryStore,
    ForkJunctionRegistry,
    MmsiBranchStatsStore,
)
from utils import haversine_distance, load_json


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


def _point_to_linestring_min_m(
    lat: float,
    lon: float,
    coords: List[List[float]],
    earth_radius: float,
) -> float:
    """LineString 坐标为 [lon, lat]；对每段采样求点到折线最小距离（米）。"""
    if len(coords) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(coords) - 1):
        lo0, la0 = float(coords[i][0]), float(coords[i][1])
        lo1, la1 = float(coords[i + 1][0]), float(coords[i + 1][1])
        for alpha in np.linspace(0.0, 1.0, 21):
            la = la0 + alpha * (la1 - la0)
            lo = lo0 + alpha * (lo1 - lo0)
            d = haversine_distance(lat, lon, la, lo, R=earth_radius)
            best = min(best, d)
    return best


class GeoJsonLineStringChannelStore(ChannelGeometryStore):
    """
    自 GeoJSON FeatureCollection 加载 LineString 中心线（如 resources/channel_centerlines.geojson）。
    每要素 properties.channel_id 缺省为 line_{idx}。
    """

    def __init__(self, path: str, earth_radius: float = 6371000.0):
        self._earth_radius = float(earth_radius)
        self._segments: Dict[str, List[List[List[float]]]] = {}
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        feats = gj.get("features") or []
        for idx, feat in enumerate(feats):
            geom = feat.get("geometry") or {}
            if geom.get("type") != "LineString":
                continue
            props = feat.get("properties") or {}
            cid = str(props.get("channel_id") or props.get("name") or f"line_{idx}")
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            self._segments.setdefault(cid, []).append(coords)

    def distance_to_centerline_m(
        self, lon: float, lat: float, channel_id: str
    ) -> float:
        lines = self._segments.get(channel_id)
        if not lines:
            return float("nan")
        dmin = float("inf")
        for coords in lines:
            dmin = min(dmin, _point_to_linestring_min_m(lat, lon, coords, self._earth_radius))
        return float(dmin) if math.isfinite(dmin) else float("nan")

    def list_channel_ids_near(
        self, lon: float, lat: float, radius_m: float
    ) -> List[str]:
        out: List[str] = []
        r = float(radius_m)
        for cid, lines in self._segments.items():
            dmin = float("inf")
            for coords in lines:
                dmin = min(dmin, _point_to_linestring_min_m(lat, lon, coords, self._earth_radius))
            if math.isfinite(dmin) and dmin <= r:
                out.append(cid)
        return out


class JsonForkJunctionRegistry(ForkJunctionRegistry):
    """
    自 JSON 加载分叉口圆域与涉及 cluster 集合，供与空间触发结合（默认 resources/fork_junctions.json）。
    """

    def __init__(self, path: str, earth_radius: float = 6371000.0):
        self._earth_radius = float(earth_radius)
        self._junctions: List[Tuple[str, float, float, float, Set[int]]] = []
        data = load_json(path)
        for j in data.get("junctions") or []:
            try:
                jid = str(j["id"])
                jlon = float(j["lon"])
                jlat = float(j["lat"])
                rad = float(j.get("radius_m", 500.0))
                cids = {int(x) for x in j.get("cluster_ids", [])}
            except (KeyError, TypeError, ValueError):
                continue
            self._junctions.append((jid, jlon, jlat, rad, cids))
        self._by_id: Dict[str, Set[int]] = {t[0]: t[4] for t in self._junctions}

    def clusters_at_junction(self, junction_id: str) -> Set[int]:
        return set(self._by_id.get(junction_id, set()))

    def junction_for_location(self, lon: float, lat: float) -> Optional[str]:
        for jid, jlon, jlat, rad, _ in self._junctions:
            d = haversine_distance(float(lat), float(lon), jlat, jlon, R=self._earth_radius)
            if d <= rad:
                return jid
        return None


def load_channel_geometry_store(
    path: Optional[str],
    earth_radius: float = 6371000.0,
) -> ChannelGeometryStore:
    if not path or not os.path.isfile(path):
        return NullChannelGeometryStore()
    try:
        store = GeoJsonLineStringChannelStore(path, earth_radius=earth_radius)
        if not store._segments:
            return NullChannelGeometryStore()
        return store
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return NullChannelGeometryStore()


def load_fork_junction_registry(
    path: Optional[str],
    earth_radius: float = 6371000.0,
) -> ForkJunctionRegistry:
    if not path or not os.path.isfile(path):
        return NullForkJunctionRegistry()
    try:
        reg = JsonForkJunctionRegistry(path, earth_radius=earth_radius)
        return reg if reg._junctions else NullForkJunctionRegistry()
    except (OSError, ValueError, TypeError, KeyError):
        return NullForkJunctionRegistry()


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
