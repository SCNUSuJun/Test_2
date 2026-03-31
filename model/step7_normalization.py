"""
步骤7：数据归一化——为LSTM模型训练准备输入数据
==============================================
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from config import NormalizationConfig, PathConfig
from utils import setup_logger, save_json, load_json, ensure_dir


class NormalizationParams:
    """单聚类 Min-Max 参数（8 个边界）。"""

    def __init__(
        self,
        lon_min: float = 0,
        lon_max: float = 1,
        lat_min: float = 0,
        lat_max: float = 1,
        sog_min: float = 0,
        sog_max: float = 1,
        cog_min: float = 0,
        cog_max: float = 1,
        cluster_id: int = 0,
    ):
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.sog_min = sog_min
        self.sog_max = sog_max
        self.cog_min = cog_min
        self.cog_max = cog_max
        self.cluster_id = cluster_id

    def to_dict(self) -> dict:
        return {
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "sog_min": self.sog_min,
            "sog_max": self.sog_max,
            "cog_min": self.cog_min,
            "cog_max": self.cog_max,
            "cluster_id": self.cluster_id,
            "schema_version": "1",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizationParams":
        return cls(
            lon_min=float(d["lon_min"]),
            lon_max=float(d["lon_max"]),
            lat_min=float(d["lat_min"]),
            lat_max=float(d["lat_max"]),
            sog_min=float(d["sog_min"]),
            sog_max=float(d["sog_max"]),
            cog_min=float(d["cog_min"]),
            cog_max=float(d["cog_max"]),
            cluster_id=int(d.get("cluster_id", 0)),
        )

    def save(self, filepath: str) -> None:
        ensure_dir(os.path.dirname(os.path.abspath(filepath)) or ".")
        save_json(self.to_dict(), filepath)

    @classmethod
    def load(cls, filepath: str) -> "NormalizationParams":
        return cls.from_dict(load_json(filepath))


class DataNormalizer:
    """按聚类类别的 Min-Max 归一化。"""

    def __init__(self, norm_config: NormalizationConfig, path_config: PathConfig):
        self.norm_cfg = norm_config
        self.path_cfg = path_config
        self.logger = setup_logger("Normalizer")

    @staticmethod
    def oriented_traj_df(
        tid: int,
        trajectories: List[pd.DataFrame],
        orientation_flip: Optional[Dict[int, bool]] = None,
    ) -> pd.DataFrame:
        """按方案 6D 方向统一：orientation_flip[tid]=True 时时间逆序。"""
        t = (
            trajectories[tid]
            .sort_values("Timestamp", kind="mergesort")
            .reset_index(drop=True)
        )
        flip = orientation_flip or {}
        if flip.get(int(tid), False):
            t = t.iloc[::-1].reset_index(drop=True)
        return t

    def compute_params_for_cluster(
        self,
        cluster_traj_indices: List[int],
        trajectories: List[pd.DataFrame],
        orientation_flip: Optional[Dict[int, bool]] = None,
    ) -> NormalizationParams:
        lons, lats, sogs, cogs = [], [], [], []
        flip = orientation_flip or {}
        for tid in cluster_traj_indices:
            t = self.oriented_traj_df(tid, trajectories, flip)
            lons.extend(t["LON"].tolist())
            lats.extend(t["LAT"].tolist())
            sogs.extend(t["SOG"].tolist())
            cogs.extend(t["COG"].tolist())
        return NormalizationParams(
            lon_min=float(np.min(lons)),
            lon_max=float(np.max(lons)),
            lat_min=float(np.min(lats)),
            lat_max=float(np.max(lats)),
            sog_min=float(np.min(sogs)),
            sog_max=float(np.max(sogs)),
            cog_min=float(np.min(cogs)),
            cog_max=float(np.max(cogs)),
        )

    def _scale(
        self, col: str, series: pd.Series, params: NormalizationParams, epsilon: float
    ) -> pd.Series:
        mapping = {
            "LON": (params.lon_min, params.lon_max),
            "LAT": (params.lat_min, params.lat_max),
            "SOG": (params.sog_min, params.sog_max),
            "COG": (params.cog_min, params.cog_max),
        }
        lo, hi = mapping[col]
        denom = hi - lo
        if abs(denom) < epsilon:
            return pd.Series(0.5, index=series.index)
        return (series - lo) / (denom + epsilon)

    def normalize_trajectory(
        self,
        traj_df: pd.DataFrame,
        params: NormalizationParams,
        epsilon: float = 1e-8,
    ) -> pd.DataFrame:
        out = traj_df.copy()
        for col in self.norm_cfg.features:
            if col in out.columns:
                out[col] = self._scale(col, out[col], params, epsilon)
        return out

    def denormalize_point(
        self,
        lon_norm: float,
        lat_norm: float,
        sog_norm: float,
        cog_norm: float,
        params: NormalizationParams,
        epsilon: float = 1e-8,
    ) -> Tuple[float, float, float, float]:
        def inv(x, lo, hi):
            d = hi - lo
            if abs(d) < epsilon:
                return lo
            return x * d + lo

        lon = inv(lon_norm, params.lon_min, params.lon_max)
        lat = inv(lat_norm, params.lat_min, params.lat_max)
        sog = inv(sog_norm, params.sog_min, params.sog_max)
        cog = inv(cog_norm, params.cog_min, params.cog_max)
        return lon, lat, sog, cog

    def normalize_cluster(
        self,
        cluster_info: Dict,
        trajectories: List[pd.DataFrame],
    ) -> Tuple[List[pd.DataFrame], NormalizationParams]:
        idxs = cluster_info["traj_indices"]
        flip = cluster_info.get("orientation_flip") or {}
        params = self.compute_params_for_cluster(idxs, trajectories, flip)
        params.cluster_id = int(cluster_info.get("cluster_id", 0))
        eps = self.norm_cfg.epsilon
        normalized = []
        for tid in idxs:
            odf = self.oriented_traj_df(tid, trajectories, flip)
            normalized.append(self.normalize_trajectory(odf, params, eps))
        return normalized, params

    def run(
        self, clusters: List[Dict], trajectories: List[pd.DataFrame]
    ) -> Dict[int, Tuple[List[pd.DataFrame], NormalizationParams]]:
        result = {}
        ensure_dir(self.path_cfg.normalization_dir)
        for c in clusters:
            cid = int(c["cluster_id"])
            ntrajs, p = self.normalize_cluster(c, trajectories)
            path = os.path.join(
                self.path_cfg.normalization_dir, f"norm_params_cluster_{cid}.json"
            )
            p.save(path)
            result[cid] = (ntrajs, p)
        return result
