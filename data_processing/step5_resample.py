"""
步骤5：轨迹数据的线性插值与等时间间隔重采样
=============================================
LON/LAT/SOG 线性；COG 圆周；Heading 由 resample_heading_mode 控制（清单 26–27）。
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from config import ResampleConfig
from utils import interpolate_cog, setup_logger


class TrajectoryResampler:
    """轨迹线性插值与等时间间隔重采样器。"""

    def __init__(self, config: ResampleConfig):
        self.cfg = config
        self.logger = setup_logger("Resampler")
        self.stats = {}

    def _ts_unix(self, ts) -> float:
        return float(pd.Timestamp(ts).timestamp())

    def resample_single_trajectory(self, traj_df: pd.DataFrame) -> pd.DataFrame:
        dt = self.cfg.resample_interval
        g = traj_df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
        if len(g) < 2:
            return g.iloc[0:0].copy()
        t0 = self._ts_unix(g.iloc[0]["Timestamp"])
        t_end = self._ts_unix(g.iloc[-1]["Timestamp"])
        if t_end <= t0:
            return g.iloc[0:0].copy()
        t_grid = np.arange(t0, t_end + 1e-9, dt)
        ts_orig = np.array([self._ts_unix(x) for x in g["Timestamp"]])
        lon_o = g["LON"].to_numpy(dtype=np.float64)
        lat_o = g["LAT"].to_numpy(dtype=np.float64)
        sog_s = pd.to_numeric(g["SOG"], errors="coerce")
        sog_o = sog_s.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
        cog_s = pd.to_numeric(g["COG"], errors="coerce")
        cog_o = cog_s.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
        head_raw = None
        h_ff = None
        if "Heading" in g.columns and self.cfg.resample_heading_mode != "omit":
            head_raw = (
                pd.to_numeric(g["Heading"], errors="coerce")
                .ffill()
                .bfill()
                .to_numpy(dtype=np.float64)
            )
            if self.cfg.resample_heading_mode == "forward_fill":
                h_series = pd.Series(head_raw, index=ts_orig).sort_index()
                h_ff = h_series.reindex(t_grid, method="ffill")
        mmsi = int(g.iloc[0]["MMSI"])
        tid = int(g.iloc[0]["TrajID"]) if "TrajID" in g.columns else 0
        dir_lab = (
            str(g.iloc[0]["direction_label"])
            if "direction_label" in g.columns
            else "unknown"
        )

        rows = []
        for j, tv in enumerate(t_grid):
            if tv < t0 or tv > t_end:
                continue
            idx = np.searchsorted(ts_orig, tv, side="right")
            if idx <= 0:
                i0, i1 = 0, 1
            elif idx >= len(ts_orig):
                i0, i1 = len(ts_orig) - 2, len(ts_orig) - 1
            else:
                i1 = idx
                i0 = idx - 1
            t_a, t_b = ts_orig[i0], ts_orig[i1]
            if t_b <= t_a:
                alpha = 0.0
            else:
                alpha = (tv - t_a) / (t_b - t_a)
            lon = lon_o[i0] + alpha * (lon_o[i1] - lon_o[i0])
            lat = lat_o[i0] + alpha * (lat_o[i1] - lat_o[i0])
            sog = sog_o[i0] + alpha * (sog_o[i1] - sog_o[i0])
            cog = interpolate_cog(float(cog_o[i0]), float(cog_o[i1]), alpha)
            rec = {
                "MMSI": mmsi,
                "TrajID": tid,
                "Timestamp": pd.to_datetime(tv, unit="s"),
                "LON": lon,
                "LAT": lat,
                "SOG": sog,
                "COG": cog,
                "direction_label": dir_lab,
            }
            if head_raw is not None:
                if self.cfg.resample_heading_mode == "circular":
                    rec["Heading"] = interpolate_cog(
                        float(head_raw[i0]), float(head_raw[i1]), alpha
                    )
                elif h_ff is not None:
                    rec["Heading"] = float(h_ff.iloc[j])
            rows.append(rec)
        return pd.DataFrame(rows)

    def resample_all(self, trajectories: List[pd.DataFrame]) -> List[pd.DataFrame]:
        out = []
        for t in trajectories:
            r = self.resample_single_trajectory(t)
            if len(r) >= 2:
                out.append(r)
        return out

    def run(self, trajectories: List[pd.DataFrame]) -> List[pd.DataFrame]:
        self.stats["input_trajs"] = len(trajectories)
        res = self.resample_all(trajectories)
        self.stats["output_trajs"] = len(res)
        return res

    def resolved_output_dir(self, base_resampled_dir: str) -> str:
        if self.cfg.output_subdir:
            return os.path.join(base_resampled_dir, self.cfg.output_subdir)
        return base_resampled_dir
