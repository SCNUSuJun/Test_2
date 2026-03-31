"""
步骤4：航行轨迹切分——将连续数据流切分为独立的航次轨迹
=====================================================
跨日连续航迹：输入须为全期合并、按 MMSI+Timestamp 排序（清单 21）。
direction_label 为 domain_heuristic（清单 23），后续可用 centerline 升级。
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from config import DomainFocusConfig, TrajSplitConfig
from config.settings import domain_focus_spatial_active
from utils import setup_logger


class TrajectorySplitter:
    """航行轨迹切分器。"""

    def __init__(
        self,
        config: TrajSplitConfig,
        domain_focus: Optional[DomainFocusConfig] = None,
    ):
        self.cfg = config
        self.domain_focus = domain_focus
        self.logger = setup_logger("TrajSplitter")
        self.stats = {}

    def _ts_unix(self, ts) -> float:
        return float(pd.Timestamp(ts).timestamp())

    def _direction_label_for_traj(self, traj: pd.DataFrame) -> str:
        """清单 23：基于起点/终点相对外海侧的启发式（LA/LB 西侧为外海近似）。"""
        if self.domain_focus is None or not domain_focus_spatial_active(
            self.domain_focus
        ):
            return "unknown"
        if traj.empty or "LON" not in traj.columns:
            return "unknown"
        lon_min, lat_min, lon_max, lat_max = self.domain_focus.roi_bbox
        sea_ref_lon = lon_min - 0.02
        lon0 = float(traj.iloc[0]["LON"])
        lon1 = float(traj.iloc[-1]["LON"])
        d0 = abs(lon0 - sea_ref_lon)
        d1 = abs(lon1 - sea_ref_lon)
        if lon1 > lon0 + 1e-4 and d0 < d1:
            return "inbound"
        if lon0 > lon1 + 1e-4 and d1 < d0:
            return "outbound"
        lat0 = float(traj.iloc[0]["LAT"])
        lat1 = float(traj.iloc[-1]["LAT"])
        if (
            lon_min <= lon0 <= lon_max
            and lon_min <= lon1 <= lon_max
            and lat_min <= lat0 <= lat_max
            and lat_min <= lat1 <= lat_max
            and abs(lon1 - lon0) < 1e-3
        ):
            return "intra-harbor"
        return "unknown"

    def split_single_vessel(
        self, vessel_df: pd.DataFrame, mmsi: int
    ) -> List[pd.DataFrame]:
        g = vessel_df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
        if g.empty:
            return []
        splits = [0]
        for k in range(1, len(g)):
            dt = self._ts_unix(g.iloc[k]["Timestamp"]) - self._ts_unix(
                g.iloc[k - 1]["Timestamp"]
            )
            if dt > self.cfg.time_gap:
                splits.append(k)
        splits.append(len(g))
        out = []
        for a, b in zip(splits[:-1], splits[1:]):
            chunk = g.iloc[a:b].copy()
            chunk["MMSI"] = mmsi
            out.append(chunk)
        return out

    def filter_short_trajectories(
        self, trajectories: List[pd.DataFrame]
    ) -> List[pd.DataFrame]:
        L = self.cfg.min_traj_length
        n0 = len(trajectories)
        kept = [t for t in trajectories if len(t) >= L]
        if self.cfg.log_short_traj_drop_ratio and n0 > 0:
            ratio = 1.0 - float(len(kept)) / float(n0)
            self.logger.info(
                "短轨迹过滤 L_min=%s：剔除条数=%s / %s，比例=%.6f",
                L,
                n0 - len(kept),
                n0,
                ratio,
            )
        return kept

    def assign_trajectory_ids(
        self, trajectories: List[pd.DataFrame]
    ) -> List[pd.DataFrame]:
        out = []
        priority = [
            "MMSI",
            "TrajID",
            "Timestamp",
            "LON",
            "LAT",
            "SOG",
            "COG",
            "direction_label",
        ]
        for tid, t in enumerate(trajectories):
            df = t.copy().reset_index(drop=True)
            df.insert(1, "TrajID", int(tid))
            df["direction_label"] = self._direction_label_for_traj(df)
            seen = set()
            cols = []
            for c in priority:
                if c in df.columns and c not in seen:
                    cols.append(c)
                    seen.add(c)
            for c in df.columns:
                if c not in seen:
                    cols.append(c)
                    seen.add(c)
            out.append(df[cols])
        return out

    def run(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        if df.empty:
            self.stats["traj_count"] = 0
            return []
        all_trajs = []
        for mmsi, g in df.groupby("MMSI", sort=False):
            all_trajs.extend(self.split_single_vessel(g, int(mmsi)))
        short_before = len(all_trajs)
        all_trajs = self.filter_short_trajectories(all_trajs)
        self.stats["after_split"] = short_before
        self.stats["after_min_len"] = len(all_trajs)
        final = self.assign_trajectory_ids(all_trajs)
        self.stats["traj_count"] = len(final)
        return final
