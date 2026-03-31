"""
步骤1：原始AIS数据的加载与字段理解
====================================
支持 ZIP 内 CSV 分块流式读取、按日落盘、子集 manifest / catalog（清单 7–9）。
"""

from __future__ import annotations

import glob
import json
import os
import re
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import DataLoadConfig, DomainFocusConfig, PathConfig, PortCleaningConfig
from config.settings import domain_focus_spatial_active
from utils import ensure_dir, load_json, save_json, setup_logger

_ZIP_DATE_RE = re.compile(r"AIS_(\d{4})_(\d{2})_(\d{2})\.zip$", re.IGNORECASE)


class AISDataLoader:
    """AIS 原始数据加载器。"""

    def __init__(
        self,
        path_config: PathConfig,
        load_config: DataLoadConfig,
        domain_config: Optional[DomainFocusConfig] = None,
        port_cleaning: Optional[PortCleaningConfig] = None,
    ):
        self.path_cfg = path_config
        self.load_cfg = load_config
        self.domain_cfg = domain_config or DomainFocusConfig()
        self.port_cleaning = port_cleaning or PortCleaningConfig()
        self.logger = setup_logger("DataLoader")

    def _raw_zip_dir(self) -> str:
        if self.domain_cfg.raw_data_root:
            return self.domain_cfg.raw_data_root
        return self.path_cfg.raw_data_dir

    def _daily_output_dir(self) -> str:
        if self.domain_cfg.subset_output_root:
            return os.path.join(
                self.domain_cfg.subset_output_root, "raw_daily_filtered"
            )
        if self.domain_cfg.enabled:
            return self.path_cfg.subsets_raw_daily_dir
        return self.path_cfg.step1_daily_output_dir

    def _metadata_dir(self) -> str:
        if self.domain_cfg.subset_output_root:
            return os.path.join(self.domain_cfg.subset_output_root, "metadata")
        return self.path_cfg.subsets_metadata_dir

    def _subset_checkpoint_path(self) -> str:
        if self.domain_cfg.subset_output_root:
            return os.path.join(
                self.domain_cfg.subset_output_root,
                "metadata",
                "subset_checkpoint.json",
            )
        return self.path_cfg.subset_checkpoint_path

    def _catalog_path(self) -> str:
        return os.path.join(self._metadata_dir(), "subset_catalog.json")

    def daily_output_path(self, date_iso: str) -> str:
        """按布局返回当日 parquet 路径。"""
        ensure_dir(self._daily_output_dir())
        if self.domain_cfg.enabled or self.domain_cfg.subset_output_root:
            y, m, d = date_iso.split("-")
            name = f"ais_lb_la_merchant_{y}_{m}_{d}.parquet"
            return os.path.join(self._daily_output_dir(), name)
        return os.path.join(self._daily_output_dir(), f"step1_{date_iso}.parquet")

    def _parse_date_from_filename(self, filename: str) -> Optional[datetime]:
        m = _ZIP_DATE_RE.search(filename)
        if not m:
            return None
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d)
        except ValueError:
            return None

    def scan_zip_files(self) -> List[str]:
        pattern = os.path.join(self._raw_zip_dir(), "AIS_*.zip")
        paths = sorted(glob.glob(pattern))
        start = self.load_cfg.date_range_start
        end = self.load_cfg.date_range_end
        if not start and not end:
            return paths
        filtered = []
        for p in paths:
            dt = self._parse_date_from_filename(os.path.basename(p))
            if dt is None:
                continue
            ds = dt.strftime("%Y-%m-%d")
            if start and ds < start:
                continue
            if end and ds > end:
                continue
            filtered.append(p)
        return filtered

    def _apply_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        cog_u = float(self.port_cleaning.cog_invalid_values[0]) if self.port_cleaning.cog_invalid_values else 360.0
        h_u = float(self.port_cleaning.heading_invalid_values[0]) if self.port_cleaning.heading_invalid_values else 511.0
        df = df.copy()
        if "COG" in df.columns:
            df.loc[df["COG"] >= cog_u, "COG"] = np.nan
        if "Heading" in df.columns:
            df.loc[pd.to_numeric(df["Heading"], errors="coerce") == h_u, "Heading"] = np.nan
        return df

    def _cheap_domain_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if not domain_focus_spatial_active(self.domain_cfg):
            return df
        out = df
        lon_min, lat_min, lon_max, lat_max = self.domain_cfg.roi_bbox
        if all(c in out.columns for c in ("LON", "LAT")):
            out = out[
                (out["LON"] >= lon_min)
                & (out["LON"] <= lon_max)
                & (out["LAT"] >= lat_min)
                & (out["LAT"] <= lat_max)
            ]
        if "VesselType" in out.columns and self.domain_cfg.vessel_type_keep:
            vt = pd.to_numeric(out["VesselType"], errors="coerce")
            keep = set(self.domain_cfg.vessel_type_keep)
            out = out[vt.isin(list(keep))]
        if (
            self.domain_cfg.transceiver_class_keep is not None
            and "TransceiverClass" in out.columns
        ):
            allowed = {str(x).strip().upper() for x in self.domain_cfg.transceiver_class_keep}
            tc = out["TransceiverClass"].astype(str).str.strip().str.upper()
            out = out[tc.isin(allowed)]
        return out

    def _finalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = list(self.load_cfg.raw_columns)
        rename = dict(self.load_cfg.column_rename)
        need = [c for c in cols if c in df.columns]
        if len(need) < 4 or "MMSI" not in df.columns:
            return pd.DataFrame()
        df = df[need].copy()
        if "BaseDateTime" in df.columns:
            # Marine Cadastre 等字段无时区后缀时按 UTC 解析；下游使用 naive datetime，语义仍为 UTC（清单 3）
            df["Timestamp"] = pd.to_datetime(
                df["BaseDateTime"],
                format=self.load_cfg.timestamp_format,
                errors="coerce",
                utc=True,
            )
            if df["Timestamp"].dt.tz is not None:
                df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
            if not self.load_cfg.keep_base_datetime_copy:
                df = df.drop(columns=["BaseDateTime"])
        elif "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
            if df["Timestamp"].dt.tz is not None:
                df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
        else:
            return pd.DataFrame()
        for old, new in rename.items():
            if old in df.columns and old != "BaseDateTime":
                df = df.rename(columns={old: new})
        df = self._apply_sentinels(df)
        df = df.dropna(subset=["MMSI", "Timestamp", "LON", "LAT"])
        ordered: List[str] = ["MMSI"]
        if "BaseDateTime" in df.columns:
            ordered.append("BaseDateTime")
        for c in ("Timestamp", "LON", "LAT", "SOG", "COG"):
            if c in df.columns:
                ordered.append(c)
        for c in df.columns:
            if c not in ordered:
                ordered.append(c)
        return df[[c for c in ordered if c in df.columns]]

    def _read_zip_csv_chunks(self, zip_path: str):
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                return
            with zf.open(names[0]) as f:
                reader = pd.read_csv(
                    f,
                    chunksize=self.load_cfg.csv_chunksize,
                    low_memory=False,
                )
                for chunk in reader:
                    yield chunk

    def load_single_zip(self, zip_path: str) -> pd.DataFrame:
        """流式读取单个 ZIP，合并为单日表。"""
        parts: List[pd.DataFrame] = []
        n_before = 0
        for chunk in self._read_zip_csv_chunks(zip_path):
            n_before += len(chunk)
            chunk2 = self._cheap_domain_filter(chunk)
            chunk3 = self._finalize_columns(chunk2)
            if not chunk3.empty:
                parts.append(chunk3)
        if not parts:
            base_cols = ["MMSI", "Timestamp", "LON", "LAT", "SOG", "COG"]
            return pd.DataFrame(columns=base_cols)
        df = pd.concat(parts, ignore_index=True)
        self.logger.debug("ZIP %s: raw_chunks_rows=%s out_rows=%s", os.path.basename(zip_path), n_before, len(df))
        return df

    def load_date_range(
        self,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
    ) -> pd.DataFrame:
        if date_start is not None:
            self.load_cfg.date_range_start = date_start
        if date_end is not None:
            self.load_cfg.date_range_end = date_end
        zips = self.scan_zip_files()
        if not zips:
            self.logger.warning("未找到匹配的 zip 文件: %s", self._raw_zip_dir())
            return pd.DataFrame(
                columns=["MMSI", "Timestamp", "LON", "LAT", "SOG", "COG"]
            )
        parts = []
        for zp in zips:
            try:
                parts.append(self.load_single_zip(zp))
            except Exception as e:
                self.logger.warning("读取失败 %s: %s", zp, e)
        if not parts:
            return pd.DataFrame(
                columns=["MMSI", "Timestamp", "LON", "LAT", "SOG", "COG"]
            )
        df = pd.concat(parts, ignore_index=True)
        return df

    def sort_by_mmsi_and_time(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.sort_values(["MMSI", "Timestamp"], kind="mergesort").reset_index(
            drop=True
        )

    def run(self) -> pd.DataFrame:
        df = self.load_date_range()
        return self.sort_by_mmsi_and_time(df)

    def _checkpoint_path(self) -> str:
        return self.path_cfg.pipeline_checkpoint_path

    def _load_pipeline_checkpoint(self) -> Dict[str, Any]:
        p = self._checkpoint_path()
        if not os.path.isfile(p):
            return {"version": 1, "zips": {}}
        try:
            data = load_json(p)
            if "zips" not in data:
                data["zips"] = {}
            return data
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "zips": {}}

    def _save_pipeline_checkpoint(self, data: Dict[str, Any]) -> None:
        ensure_dir(os.path.dirname(os.path.abspath(self._checkpoint_path())) or ".")
        save_json(data, self._checkpoint_path())

    def _load_subset_checkpoint(self) -> Dict[str, Any]:
        p = self._subset_checkpoint_path()
        if not os.path.isfile(p):
            return {"version": 1, "dates": {}}
        try:
            data = load_json(p)
            if "dates" not in data:
                data["dates"] = {}
            return data
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "dates": {}}

    def _save_subset_checkpoint(self, data: Dict[str, Any]) -> None:
        ensure_dir(os.path.dirname(os.path.abspath(self._subset_checkpoint_path())) or ".")
        save_json(data, self._subset_checkpoint_path())

    def _build_manifest(
        self,
        zip_base: str,
        date_iso: str,
        rows_before: int,
        rows_after: int,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        m: Dict[str, Any] = {
            "schema_version": "1",
            "original_zip": zip_base,
            "date_iso": date_iso,
            "rows_before_filter": int(rows_before),
            "rows_after_filter": int(rows_after),
            "study_name": self.domain_cfg.study_name,
            "roi_version": self.domain_cfg.roi_version,
            "filter_profile": self.domain_cfg.filter_profile,
        }
        if df.empty:
            m["time_min"] = None
            m["time_max"] = None
            m["vessel_type_dist"] = {}
            m["transceiver_class_dist"] = {}
            m["status_dist"] = {}
            return m
        m["time_min"] = str(pd.Timestamp(df["Timestamp"].min()))
        m["time_max"] = str(pd.Timestamp(df["Timestamp"].max()))
        if "VesselType" in df.columns:
            vc = df["VesselType"].value_counts(dropna=False)
            m["vessel_type_dist"] = {str(k): int(v) for k, v in vc.items()}
        else:
            m["vessel_type_dist"] = {}
        if "TransceiverClass" in df.columns:
            tc = df["TransceiverClass"].astype(str).value_counts(dropna=False)
            m["transceiver_class_dist"] = {str(k): int(v) for k, v in tc.items()}
        else:
            m["transceiver_class_dist"] = {}
        if "Status" in df.columns:
            st = df["Status"].astype(str).value_counts(dropna=False)
            m["status_dist"] = {str(k): int(v) for k, v in st.items()}
        else:
            m["status_dist"] = {}
        return m

    def _update_catalog(self, date_iso: str, manifest_rel: str) -> None:
        ensure_dir(self._metadata_dir())
        cat_path = self._catalog_path()
        if os.path.isfile(cat_path):
            try:
                cat = load_json(cat_path)
            except (OSError, json.JSONDecodeError):
                cat = {}
        else:
            cat = {}
        if "dates" not in cat:
            cat["dates"] = {}
        cat["dates"][date_iso] = {"manifest": manifest_rel}
        cat["schema_version"] = "1"
        save_json(cat, cat_path)

    def run_with_checkpoint(self) -> pd.DataFrame:
        """
        按 zip 逐日写入 parquet；支持 pipeline_checkpoint 与子集 checkpoint 跳过已完成。
        """
        ensure_dir(self._daily_output_dir())
        ensure_dir(self._metadata_dir())
        ck_pipe = self._load_pipeline_checkpoint()
        ck_sub = self._load_subset_checkpoint()
        zips_meta = ck_pipe.setdefault("zips", {})
        dates_meta = ck_sub.setdefault("dates", {})

        zips = self.scan_zip_files()
        for zp in zips:
            base = os.path.basename(zp)
            try:
                st = os.stat(zp)
            except OSError:
                continue
            dt = self._parse_date_from_filename(base)
            if dt is None:
                continue
            date_iso = dt.strftime("%Y-%m-%d")
            out_parquet = self.daily_output_path(date_iso)
            manifest_name = f"subset_manifest_{date_iso}.json"
            manifest_path = os.path.join(self._metadata_dir(), manifest_name)

            prev = zips_meta.get(base)
            sub_prev = dates_meta.get(date_iso)
            if (
                prev
                and prev.get("size") == st.st_size
                and prev.get("mtime") == int(st.st_mtime)
                and os.path.isfile(out_parquet)
                and sub_prev
                and sub_prev.get("parquet") == os.path.basename(out_parquet)
                and os.path.isfile(manifest_path)
            ):
                continue

            rows_before = 0
            parts: List[pd.DataFrame] = []
            try:
                for chunk in self._read_zip_csv_chunks(zp):
                    rows_before += len(chunk)
                    chunk2 = self._cheap_domain_filter(chunk)
                    chunk3 = self._finalize_columns(chunk2)
                    if not chunk3.empty:
                        parts.append(chunk3)
            except Exception as e:
                self.logger.warning("按日流式读取失败 %s: %s", zp, e)
                continue

            if not parts:
                day_df = pd.DataFrame(
                    columns=["MMSI", "Timestamp", "LON", "LAT", "SOG", "COG"]
                )
            else:
                day_df = pd.concat(parts, ignore_index=True)
                day_df = self.sort_by_mmsi_and_time(day_df)

            rows_after = len(day_df)
            ensure_dir(os.path.dirname(os.path.abspath(out_parquet)) or ".")
            day_df.to_parquet(out_parquet, index=False)

            manifest = self._build_manifest(base, date_iso, rows_before, rows_after, day_df)
            save_json(manifest, manifest_path)

            zips_meta[base] = {
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "rows": rows_after,
                "parquet": os.path.basename(out_parquet),
                "date_iso": date_iso,
            }
            dates_meta[date_iso] = {
                "parquet": os.path.basename(out_parquet),
                "manifest": manifest_name,
            }
            self._update_catalog(date_iso, manifest_name)

        self._save_pipeline_checkpoint(ck_pipe)
        self._save_subset_checkpoint(ck_sub)

        allowed_dates = set()
        for zp in zips:
            dt = self._parse_date_from_filename(os.path.basename(zp))
            if dt is not None:
                allowed_dates.add(dt.strftime("%Y-%m-%d"))

        parquet_files = sorted(glob.glob(os.path.join(self._daily_output_dir(), "*.parquet")))
        if allowed_dates:
            filtered = []
            for pf in parquet_files:
                bn = os.path.basename(pf)
                if self.domain_cfg.enabled or self.domain_cfg.subset_output_root:
                    if not bn.startswith("ais_lb_la_merchant_"):
                        continue
                    core = bn.replace("ais_lb_la_merchant_", "").replace(".parquet", "")
                    parts_d = core.split("_")
                    if len(parts_d) == 3:
                        d_iso = f"{parts_d[0]}-{int(parts_d[1]):02d}-{int(parts_d[2]):02d}"
                else:
                    if not bn.startswith("step1_"):
                        continue
                    d_iso = bn[6:-8]
                if d_iso in allowed_dates:
                    filtered.append(pf)
            parquet_files = filtered
        if not parquet_files:
            self.logger.warning("未找到任何按日 parquet，请检查 zip 与日期范围")
            return pd.DataFrame(
                columns=["MMSI", "Timestamp", "LON", "LAT", "SOG", "COG"]
            )
        parts_read: List[pd.DataFrame] = []
        for pf in parquet_files:
            try:
                parts_read.append(pd.read_parquet(pf))
            except Exception as e:
                self.logger.warning("读取 parquet 失败 %s: %s", pf, e)
        if not parts_read:
            return pd.DataFrame(
                columns=["MMSI", "Timestamp", "LON", "LAT", "SOG", "COG"]
            )
        df = pd.concat(parts_read, ignore_index=True)
        return self.sort_by_mmsi_and_time(df)
