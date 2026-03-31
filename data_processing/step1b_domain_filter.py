"""
步骤1b：LA/LB 业务域过滤（清单 5）
================================
船型 / TransceiverClass 可在步骤1 已做；此处负责精筛 polygon、主航路 corridor、
terminal/anchorage exclusion。不实现异常值五规则（见 step2）。
空间顺序：outer ROI polygon → inner ROI（若配置）→ corridor（若配置）→ 排除区（清单 12）。
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

import pandas as pd

from config import DomainFocusConfig
from config.settings import domain_focus_spatial_active

try:
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "step1b_domain_filter 需要 shapely，请安装: pip install shapely"
    ) from e

from utils import setup_logger


def _load_unary_geometry(geojson_path: str) -> Optional[Any]:
    if not geojson_path or not os.path.isfile(geojson_path):
        return None
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    geoms: List[Any] = []
    t = data.get("type")
    if t == "FeatureCollection":
        for feat in data.get("features", []):
            g = feat.get("geometry")
            if g:
                geoms.append(shape(g))
    elif t == "Feature" and data.get("geometry"):
        geoms.append(shape(data["geometry"]))
    elif "coordinates" in data:
        geoms.append(shape(data))
    if not geoms:
        return None
    return unary_union(geoms)


def _load_many_union(paths: List[str]) -> Optional[Any]:
    parts: List[Any] = []
    for p in paths:
        g = _load_unary_geometry(p)
        if g is not None:
            parts.append(g)
    if not parts:
        return None
    return unary_union(parts)


class DomainSpatialFilter:
    """基于 GeoJSON 的空间过滤（与 DomainFocusConfig 路径字段对齐）。"""

    def __init__(self, config: DomainFocusConfig):
        self.cfg = config
        self.logger = setup_logger("DomainSpatialFilter")
        self._roi = _load_unary_geometry(config.roi_polygon_path)
        self._inner = _load_unary_geometry(
            (config.inner_roi_polygon_path or "").strip()
        )
        self._corridor = _load_unary_geometry(config.route_corridor_path)
        self._exclude_terminal = _load_many_union(config.exclude_terminal_polygon_paths)
        self._exclude_anchor = _load_many_union(config.exclude_anchorage_polygon_paths)

    def _point_in(self, lon: float, lat: float, geom: Any) -> bool:
        if geom is None:
            return True
        try:
            return bool(geom.covers(Point(float(lon), float(lat))))
        except Exception:
            return False

    def _point_outside(self, lon: float, lat: float, geom: Any) -> bool:
        if geom is None:
            return True
        try:
            return not geom.intersects(Point(float(lon), float(lat)))
        except Exception:
            return True

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not domain_focus_spatial_active(self.cfg):
            return df
        if all(
            x is None
            for x in (
                self._roi,
                self._inner,
                self._corridor,
                self._exclude_terminal,
                self._exclude_anchor,
            )
        ):
            self.logger.info("步骤1b：未配置 GeoJSON 路径，跳过精筛")
            return df

        out = df.copy()
        n0 = len(out)
        if self._roi is not None:
            m = out.apply(
                lambda r: self._point_in(r["LON"], r["LAT"], self._roi), axis=1
            )
            out = out.loc[m].copy()
        if self._inner is not None:
            mi = out.apply(
                lambda r: self._point_in(r["LON"], r["LAT"], self._inner), axis=1
            )
            out = out.loc[mi].copy()
        if self._corridor is not None:
            m2 = out.apply(
                lambda r: self._point_in(r["LON"], r["LAT"], self._corridor), axis=1
            )
            out = out.loc[m2].copy()
        if self._exclude_terminal is not None:
            m3 = out.apply(
                lambda r: self._point_outside(
                    r["LON"], r["LAT"], self._exclude_terminal
                ),
                axis=1,
            )
            out = out.loc[m3].copy()
        if self._exclude_anchor is not None:
            m4 = out.apply(
                lambda r: self._point_outside(
                    r["LON"], r["LAT"], self._exclude_anchor
                ),
                axis=1,
            )
            out = out.loc[m4].copy()
        self.logger.info("步骤1b：空间精筛 %s → %s 行", n0, len(out))
        return out.reset_index(drop=True)
