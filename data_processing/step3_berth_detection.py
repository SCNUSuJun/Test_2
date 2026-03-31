"""
步骤3：停泊点检测与停泊数据段删除
====================================
支持 mode=delete（方案原语义）与 label_only（清单 17–20，先标注再验收）。
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import BerthConfig
from utils import haversine_distance, setup_logger

try:
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union
except ImportError:  # pragma: no cover
    Point = None  # type: ignore
    shape = None  # type: ignore
    unary_union = None  # type: ignore


def _load_unary_geometry(geojson_path: str) -> Optional[Any]:
    if not geojson_path or not os.path.isfile(geojson_path) or shape is None:
        return None
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    geoms: List[Any] = []
    if data.get("type") == "FeatureCollection":
        for feat in data.get("features", []):
            g = feat.get("geometry")
            if g:
                geoms.append(shape(g))
    elif data.get("type") == "Feature" and data.get("geometry"):
        geoms.append(shape(data["geometry"]))
    if not geoms:
        return None
    return unary_union(geoms)


def _load_many_union(paths: List[str]) -> Optional[Any]:
    parts: List[Any] = []
    for p in paths:
        g = _load_unary_geometry(p)
        if g is not None:
            parts.append(g)
    if not parts or unary_union is None:
        return None
    return unary_union(parts)


class BerthDetector:
    """停泊数据段检测器。"""

    def __init__(self, config: BerthConfig):
        self.cfg = config
        self.logger = setup_logger("BerthDetector")
        self.stats = {}
        self._terminal_union = _load_many_union(config.terminal_polygon_paths)
        self._anchorage_union = _load_many_union(config.anchorage_polygon_paths)
        self._corridor = _load_unary_geometry(
            config.channel_corridor_polygon_path or ""
        )

    def _ts_unix(self, ts) -> float:
        return float(pd.Timestamp(ts).timestamp())

    def detect_berth_segments_single_vessel(
        self, vessel_df: pd.DataFrame
    ) -> List[Tuple[int, int]]:
        """返回应删除的闭区间 [start, end]（reset 后的行号）。"""
        cfg = self.cfg
        g = vessel_df.reset_index(drop=True)
        n = len(g)
        to_drop: Set[int] = set()
        i = 0
        while i < n:
            j = i + 1
            while j < n:
                row_i = g.iloc[i]
                row_j = g.iloc[j]
                d_ij = haversine_distance(
                    float(row_i["LAT"]),
                    float(row_i["LON"]),
                    float(row_j["LAT"]),
                    float(row_j["LON"]),
                )
                if d_ij >= cfg.distance_threshold:
                    break
                sogs = pd.to_numeric(g.iloc[i : j + 1]["SOG"], errors="coerce")
                if (sogs >= cfg.speed_threshold).fillna(False).any():
                    break
                j += 1
            j_last = j - 1
            if j_last < i:
                i += 1
                continue
            span = self._ts_unix(g.iloc[j_last]["Timestamp"]) - self._ts_unix(
                g.iloc[i]["Timestamp"]
            )
            if span > cfg.time_threshold:
                for k in range(i, j_last + 1):
                    to_drop.add(k)
                i = j
            else:
                i = i + 1
        if not to_drop:
            return []
        sorted_idx = sorted(to_drop)
        segments = []
        s = sorted_idx[0]
        prev = s
        for x in sorted_idx[1:]:
            if x == prev + 1:
                prev = x
            else:
                segments.append((s, prev))
                s = x
                prev = x
        segments.append((s, prev))
        return segments

    def remove_berth_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_global = []
        for _, g in df.groupby("MMSI", sort=False):
            g = g.sort_values("Timestamp", kind="mergesort")
            idx = g.index.to_numpy()
            g_reset = g.reset_index(drop=True)
            segs = self.detect_berth_segments_single_vessel(g_reset)
            for a, b in segs:
                for k in range(a, b + 1):
                    drop_global.append(idx[k])
        if not drop_global:
            return df.copy()
        return (
            df.drop(index=drop_global)
            .sort_values(["MMSI", "Timestamp"], kind="mergesort")
            .reset_index(drop=True)
        )

    def _point_in_geom(self, lon: float, lat: float, geom: Any) -> bool:
        if geom is None or Point is None:
            return False
        try:
            return bool(geom.covers(Point(float(lon), float(lat))))
        except Exception:
            return False

    def _run_label_only(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        n = len(out)
        term = np.zeros(n, dtype=bool)
        anch = np.zeros(n, dtype=bool)
        ch = np.zeros(n, dtype=bool)
        lons = out["LON"].to_numpy(dtype=np.float64)
        lats = out["LAT"].to_numpy(dtype=np.float64)
        sog = pd.to_numeric(out["SOG"], errors="coerce").to_numpy(dtype=np.float64)
        for i in range(n):
            lon, lat = float(lons[i]), float(lats[i])
            sg = sog[i]
            if self._terminal_union is not None and self._point_in_geom(
                lon, lat, self._terminal_union
            ):
                if not np.isnan(sg) and sg <= self.cfg.label_terminal_sog_max:
                    term[i] = True
            if self._anchorage_union is not None and self._point_in_geom(
                lon, lat, self._anchorage_union
            ):
                if not np.isnan(sg) and sg <= self.cfg.label_anchorage_sog_max:
                    anch[i] = True
            if self._corridor is not None and self._point_in_geom(
                lon, lat, self._corridor
            ):
                if not np.isnan(sg) and sg <= self.cfg.channel_low_speed_for_label:
                    ch[i] = True
        out["is_terminal_dwell"] = term
        out["is_anchorage_wait"] = anch
        out["is_channel_low_speed"] = ch
        self.stats["input"] = len(df)
        self.stats["output"] = len(out)
        self.stats["mode"] = "label_only"
        return out

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.mode == "label_only":
            return self._run_label_only(df)
        self.stats["input"] = len(df)
        out = self.remove_berth_segments(df)
        self.stats["output"] = len(out)
        self.stats["mode"] = "delete"
        return out
