"""
在线预测全链路脚本（步骤10 → 步骤12）
=======================================
"""

import sys
import os
import time
import json
import argparse
import urllib.request
import urllib.error
from typing import List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config, RSTPMConfig
from online import RealtimePreprocessor, ClusterMatcher, TrajectoryPredictor
from resources import (
    InMemoryMmsiBranchStatsStore,
    NullChannelGeometryStore,
    NullForkJunctionRegistry,
    prune_branch_predictions_by_geometry,
)
from utils import setup_logger


def _parse_api_payload(raw: bytes) -> List[dict]:
    """支持 JSON 数组或 NDJSON。"""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


class OnlinePredictionSystem:
    """整合步骤10/11/12。"""

    def __init__(self, config: RSTPMConfig):
        self.config = config
        self.logger = setup_logger("OnlineSystem", config.paths.log_dir)
        self.branch_stats = InMemoryMmsiBranchStatsStore()
        self.preprocessor = RealtimePreprocessor(
            filter_config=config.filter,
            resample_config=config.resample,
            predict_config=config.predict,
            max_T=int(config.predict.buffer_capacity_steps),
        )
        self.matcher = ClusterMatcher(
            config.predict,
            geometry_store=NullChannelGeometryStore(),
            branch_stats_store=self.branch_stats,
            junction_registry=NullForkJunctionRegistry(),
            device=config.device,
        )
        self.predictor = TrajectoryPredictor(
            predict_config=config.predict,
            resample_config=config.resample,
            device=config.device,
        )
    def load_offline_assets(self) -> None:
        self.logger.info("加载离线训练产物...")
        self.matcher.load_cluster_assets(
            cluster_dir=self.config.paths.cluster_dir,
            model_dir=self.config.paths.model_dir,
            norm_dir=self.config.paths.normalization_dir,
        )
        self.logger.info("已加载 %s 个聚类模型", len(self.matcher.cluster_assets))
        mc_path = os.path.join(
            self.config.paths.cluster_dir, "mmsi_cluster_counts.json"
        )
        n_mc = self.branch_stats.load_from_offline_json(mc_path)
        if n_mc:
            self.logger.info("已加载 MMSI×cluster 历史计数键数: %s", n_mc)
        else:
            self.logger.warning(
                "未找到或无法解析 mmsi_cluster_counts.json，分叉概率将退化为均匀（需先跑离线流水线）"
            )
        max_t_assets = 0
        if self.matcher.cluster_assets:
            max_t_assets = max(int(a.T) for a in self.matcher.cluster_assets.values())
        cap = int(self.config.predict.buffer_capacity_steps)
        merged = max(cap, max_t_assets)
        self.preprocessor.set_buffer_capacity(merged)
        self.logger.info(
            "RealtimePreprocessor 缓冲容量=%s（buffer_capacity_steps=%s, max(簇T)=%s）",
            merged,
            cap,
            max_t_assets,
        )

    def seed_branch_stats_from_csv(self, csv_path: str) -> None:
        """用历史 CSV 粗略填充 MMSI→cluster 计数（需含 cluster_id 或 TrajID 等列时可扩展）。"""
        if not os.path.isfile(csv_path):
            return
        try:
            df = pd.read_csv(csv_path, nrows=50000, low_memory=False)
        except OSError:
            return
        if "cluster_id" in df.columns and "MMSI" in df.columns:
            for (mmsi, cid), g in df.groupby(["MMSI", "cluster_id"]):
                self.branch_stats.seed_from_counts(int(mmsi), {int(cid): len(g)})

    def predict_for_vessel(self, mmsi: int, current_timestamp: float) -> dict:
        buf = self.preprocessor.get_vessel_buffer(mmsi)
        if buf is None:
            return {"status": "no_buffer", "mmsi": mmsi}

        fork_cands = self.matcher.detect_fork_situation(buf)
        P = self.config.predict.prediction_steps

        if fork_cands is not None and len(fork_cands) >= 2:
            cids = [c["cluster_id"] for c in fork_cands]
            probs = self.matcher.compute_fork_probabilities(mmsi, cids)
            branch = self.predictor.predict_with_fork(
                fork_cands, buf, current_timestamp
            )
            last = buf.get_latest_point()
            if last is not None:
                _, obs_lon, obs_lat, _, _ = last
                branch = prune_branch_predictions_by_geometry(
                    self.matcher.geometry_store,
                    obs_lon,
                    obs_lat,
                    branch,
                    self.config.predict.fork_deviation_threshold,
                )
            primary = fork_cands[0]["cluster_id"]
            primary_preds = branch.get(primary, [])
            return self.predictor.format_prediction_output(
                mmsi,
                primary,
                primary_preds,
                fork_probabilities=probs,
                is_fork=True,
                branch_predictions=branch,
            )

        match = self.matcher.match_single_vessel(buf)
        if match.get("is_unknown"):
            return {
                "status": "unknown_cluster",
                "mmsi": mmsi,
                "match_distance": match.get("match_distance"),
            }
        preds = self.predictor.predict_multi_step(
            match["model"],
            buf,
            match["norm_params"],
            match["T"],
            P,
            current_timestamp,
        )
        return self.predictor.format_prediction_output(
            mmsi, match["cluster_id"], preds, is_fork=False
        )

    def process_ais_message(
        self,
        mmsi: int,
        timestamp: float,
        lon: float,
        lat: float,
        sog: float,
        cog: float,
    ) -> dict:
        ready = self.preprocessor.process_message(
            mmsi, timestamp, lon, lat, sog, cog
        )
        if ready is not None:
            return self.predict_for_vessel(ready, float(timestamp))
        return {"status": "buffering", "mmsi": mmsi}

    def run_simulate_mode(self, data_file: str) -> None:
        self.seed_branch_stats_from_csv(data_file)
        df = pd.read_csv(data_file, low_memory=False)
        if "BaseDateTime" in df.columns and "Timestamp" not in df.columns:
            df = df.rename(columns={"BaseDateTime": "Timestamp"})
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        df = df.sort_values("Timestamp", kind="mergesort")
        for _, row in df.iterrows():
            ts = row["Timestamp"].timestamp()
            r = self.process_ais_message(
                int(row["MMSI"]),
                ts,
                float(row["LON"]),
                float(row["LAT"]),
                float(row["SOG"]),
                float(row["COG"]),
            )
            if r.get("points") or r.get("branch_predictions"):
                self.logger.info(
                    "MMSI %s 预测: keys=%s", row["MMSI"], list(r.keys())
                )
                break

    def run_realtime_mode(
        self,
        api_url: str,
        poll_interval_sec: float = 5.0,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        使用 urllib 周期性 GET，响应体为 JSON 数组或 NDJSON。
        每条记录字段：MMSI, Timestamp(Unix 或 ISO), LON, LAT, SOG, COG
        """
        n = 0
        while max_iterations is None or n < max_iterations:
            try:
                with urllib.request.urlopen(api_url, timeout=30) as resp:
                    payload = _parse_api_payload(resp.read())
            except urllib.error.URLError as e:
                self.logger.warning("拉取失败: %s", e)
                time.sleep(poll_interval_sec)
                n += 1
                continue
            for rec in payload:
                mmsi = int(rec["MMSI"])
                ts = rec["Timestamp"]
                if isinstance(ts, str):
                    ts = pd.Timestamp(ts).timestamp()
                else:
                    ts = float(ts)
                out = self.process_ais_message(
                    mmsi,
                    ts,
                    float(rec["LON"]),
                    float(rec["LAT"]),
                    float(rec["SOG"]),
                    float(rec["COG"]),
                )
                if out.get("points") or out.get("branch_predictions"):
                    self.logger.info("预测输出: %s", out.get("cluster_id"))
            time.sleep(poll_interval_sec)
            n += 1


def main():
    parser = argparse.ArgumentParser(description="RSTPM 在线预测系统")
    parser.add_argument(
        "--mode", type=str, default="simulate", choices=["simulate", "realtime"]
    )
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--api_url", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--realtime_poll_sec", type=float, default=5.0, help="实时模式轮询间隔秒"
    )
    parser.add_argument(
        "--realtime_max_iter",
        type=int,
        default=None,
        help="实时模式最大轮数（默认无限，调试时可设 1）",
    )
    args = parser.parse_args()

    config = get_default_config()
    config.device = args.device

    system = OnlinePredictionSystem(config)
    system.load_offline_assets()

    if args.mode == "simulate":
        assert args.data_file, "模拟模式需要指定 --data_file"
        system.run_simulate_mode(args.data_file)
    else:
        assert args.api_url, "实时模式需要指定 --api_url"
        system.run_realtime_mode(
            args.api_url,
            poll_interval_sec=args.realtime_poll_sec,
            max_iterations=args.realtime_max_iter,
        )


if __name__ == "__main__":
    main()
