"""
步骤2：逐条AIS报文的异常值检测与过滤
======================================
Marine Cadastre：COG=360、Heading=511 置空且不因 COG 缺失删行（domain_adapted，见 LB_LA_ADAPTATION_PLAN.md）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from config import FilterConfig
from utils import compute_implied_speed, setup_logger

if TYPE_CHECKING:
    from config import PortCleaningConfig


class AnomalyFilter:
    """AIS 数据异常值过滤器。"""

    def __init__(
        self,
        config: FilterConfig,
        port_cleaning: Optional["PortCleaningConfig"] = None,
    ):
        self.cfg = config
        self.port_cleaning = port_cleaning
        self.logger = setup_logger("AnomalyFilter")
        self.stats = {}

    @staticmethod
    def _ts_unix(ts) -> float:
        return float(pd.Timestamp(ts).timestamp())

    def apply_invalid_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        """清单 13：COG/Heading 不可用编码 → NaN。"""
        c = self.cfg
        out = df.copy()
        if "COG" in out.columns:
            out.loc[out["COG"] >= float(c.cog_unavailable), "COG"] = np.nan
        if "Heading" in out.columns:
            h = pd.to_numeric(out["Heading"], errors="coerce")
            out.loc[h == float(c.heading_unavailable), "Heading"] = np.nan
        return out

    def filter_invalid_lat_lon_time_mmsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """清单 13：非法 LAT/LON、时间解析失败、MMSI 非法 → 删除。"""
        m = (
            df["LON"].between(-180.0, 180.0)
            & df["LAT"].between(-90.0, 90.0)
            & df["Timestamp"].notna()
        )
        mmsi = pd.to_numeric(df["MMSI"], errors="coerce")
        m = m & mmsi.notna() & (mmsi >= 100000000) & (mmsi <= 999999999)
        return df.loc[m].copy()

    def filter_geo_range(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        m = (
            (df["LON"] >= c.lon_min)
            & (df["LON"] <= c.lon_max)
            & (df["LAT"] >= c.lat_min)
            & (df["LAT"] <= c.lat_max)
        )
        return df.loc[m].copy()

    def _log_sog_threshold_candidates(self, df: pd.DataFrame) -> None:
        """清单 14：记录多档航速上限下的删除比例（相对当前输入行）。"""
        if self.port_cleaning is None or not self.port_cleaning.sog_max_knots_candidates:
            return
        c = self.cfg
        sog = pd.to_numeric(df["SOG"], errors="coerce")
        n = max(len(df), 1)
        for cand in self.port_cleaning.sog_max_knots_candidates:
            keep = sog.notna() & (sog >= c.sog_min) & (sog <= float(cand))
            drop_ratio = 1.0 - float(keep.sum()) / float(n)
            removed = df.loc[~keep]
            mmsi_n = (
                int(removed["MMSI"].nunique())
                if "MMSI" in df.columns and len(removed) > 0
                else 0
            )
            self.logger.info(
                "SOG 候选阈值 %.1f kn：相对当前步输入删除比例=%.6f，涉及不同 MMSI 数=%s",
                cand,
                drop_ratio,
                mmsi_n,
            )

    def filter_sog(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        self._log_sog_threshold_candidates(df)
        sog = pd.to_numeric(df["SOG"], errors="coerce")
        m = sog.notna() & (sog >= c.sog_min) & (sog <= c.sog_max)
        return df.loc[m].copy()

    def filter_cog(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        cog = pd.to_numeric(df["COG"], errors="coerce")
        if c.allow_missing_cog_rows:
            bad = cog.notna() & ((cog < c.cog_min) | (cog >= c.cog_max))
            return df.loc[~bad].copy()
        m = cog.notna() & (cog >= c.cog_min) & (cog < c.cog_max)
        return df.loc[m].copy()

    def _filter_jump_single_group(self, g: pd.DataFrame) -> pd.DataFrame:
        """方案 2.6：删除后一点并继续与下一点比较。"""
        c = self.cfg
        v_thresh = c.sog_max * c.knot_to_ms * c.jump_speed_factor
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 2:
            return g
        out_idx = [0]
        i = 0
        j = 1
        while j < n:
            row_i = g.iloc[out_idx[i]]
            row_j = g.iloc[j]
            t_i = self._ts_unix(row_i["Timestamp"])
            t_j = self._ts_unix(row_j["Timestamp"])
            dt = t_j - t_i
            if dt <= 0:
                j += 1
                continue
            v_imp = compute_implied_speed(
                float(row_i["LAT"]),
                float(row_i["LON"]),
                t_i,
                float(row_j["LAT"]),
                float(row_j["LON"]),
                t_j,
                R=c.earth_radius,
            )
            if v_imp > v_thresh:
                j += 1
                continue
            out_idx.append(j)
            i += 1
            j += 1
        return g.iloc[out_idx].copy()

    def filter_jump_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for _, g in df.groupby("MMSI", sort=False):
            g = g.sort_values("Timestamp", kind="mergesort")
            parts.append(self._filter_jump_single_group(g))
        if not parts:
            return df.iloc[0:0].copy()
        return pd.concat(parts, ignore_index=True)

    def filter_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        方案 2.7 / .cursorrules：相邻点时间差 < duplicate_time_threshold 秒则丢弃后点。
        清单 13「经纬重复且时间几乎相同」在静止船同坐标、同秒级报告场景下已由该规则覆盖；
        不得改为「仅同坐标去重」以免与 .cursorrules 重复点语义漂移。
        """
        th = self.cfg.duplicate_time_threshold
        parts = []
        for _, g in df.groupby("MMSI", sort=False):
            g = g.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
            if g.empty:
                continue
            keep = [0]
            last_ts = self._ts_unix(g.iloc[0]["Timestamp"])
            for i in range(1, len(g)):
                ts = self._ts_unix(g.iloc[i]["Timestamp"])
                if ts - last_ts < th:
                    continue
                keep.append(i)
                last_ts = ts
            parts.append(g.iloc[keep].copy())
        if not parts:
            return df.iloc[0:0].copy()
        return pd.concat(parts, ignore_index=True)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        self.stats = {}
        n0 = len(df)
        d0 = self.filter_invalid_lat_lon_time_mmsi(df)
        self.stats["after_basic"] = len(d0)
        d1 = self.filter_geo_range(d0)
        self.stats["after_geo"] = len(d1)
        d2 = self.filter_sog(d1)
        self.stats["after_sog"] = len(d2)
        d2b = self.apply_invalid_sentinels(d2)
        d3 = self.filter_cog(d2b)
        self.stats["after_cog"] = len(d3)
        d4 = self.filter_jump_distance(d3)
        self.stats["after_jump"] = len(d4)
        d5 = self.filter_duplicates(d4)
        self.stats["after_dup"] = len(d5)
        self.stats["input_rows"] = n0
        return d5

    def get_stats(self) -> dict:
        return self.stats
