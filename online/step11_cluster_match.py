"""
步骤11：在线阶段——确定船舶所属的聚类类别并选择对应的预测模型
==============================================================
"""

import os
import re
import glob
import json
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import PredictConfig
from resources import (
    ChannelGeometryStore,
    ForkJunctionRegistry,
    MmsiBranchStatsStore,
    NullForkJunctionRegistry,
)
from schemas import ClusterAsset
from clustering.step6c_dtw_refine import ImprovedDTW
from model.step7_normalization import NormalizationParams
from model.step9_lstm_train import LSTMTrainer
from online.step10_realtime_preprocess import VesselBuffer
from utils import setup_logger, load_pickle, load_json


_NORM_CID_RE = re.compile(r"norm_params_cluster_(\d+)\.json$")


def _load_cluster_from_bundle(
    bundle_path: str,
    model_dir: str,
    trainer: LSTMTrainer,
) -> Optional[Tuple[int, ClusterAsset]]:
    try:
        bd = load_json(bundle_path)
    except (OSError, json.JSONDecodeError):
        return None
    cid = int(bd["cluster_id"])
    weights = bd.get("weights_path", "")
    norm_p = bd.get("norm_params_path", "")
    rep_p = bd.get("rep_traj_path", "")
    if not os.path.isfile(weights):
        w2 = os.path.join(model_dir, os.path.basename(weights))
        if os.path.isfile(w2):
            weights = w2
    if not os.path.isfile(weights):
        return None
    if not os.path.isfile(norm_p):
        return None
    if not os.path.isfile(rep_p):
        return None
    params = NormalizationParams.load(norm_p)
    model, _cfg = trainer.load_model(weights)
    z = np.load(rep_p, allow_pickle=False)
    flon = np.asarray(z["lon"], dtype=np.float64)
    flat = np.asarray(z["lat"], dtype=np.float64)
    T = int(bd.get("T", _cfg["T"]))
    H = int(bd.get("H", _cfg["H"]))
    asset = ClusterAsset(
        cluster_id=cid,
        feature_traj_lon=flon,
        feature_traj_lat=flat,
        T=T,
        H=H,
        norm_params=params,
        model=model,
        model_path=weights,
        norm_params_path=norm_p,
        rep_traj_path=rep_p,
    )
    return cid, asset


class ClusterMatcher:
    """在线聚类匹配与资产加载。"""

    def __init__(
        self,
        predict_config: PredictConfig,
        geometry_store: Optional[ChannelGeometryStore] = None,
        branch_stats_store: Optional[MmsiBranchStatsStore] = None,
        junction_registry: Optional[ForkJunctionRegistry] = None,
        device: str = "cpu",
    ):
        self.predict_cfg = predict_config
        self.logger = setup_logger("ClusterMatcher")
        self.dtw = ImprovedDTW()
        self.geometry_store = geometry_store
        self.branch_stats_store = branch_stats_store
        self.junction_registry = junction_registry
        self.cluster_assets: Dict[int, ClusterAsset] = {}
        self._device = device
        # unknown_threshold 为 None 时，在 load_cluster_assets 中解析为标定值或 fallback
        self._effective_unknown_threshold: float = float(
            predict_config.unknown_threshold
            if predict_config.unknown_threshold is not None
            else predict_config.unknown_threshold_fallback
        )

    def load_cluster_assets(
        self, cluster_dir: str, model_dir: str, norm_dir: str
    ) -> None:
        from config import TrainConfig, PathConfig

        self.cluster_assets.clear()
        trainer = LSTMTrainer(TrainConfig(), PathConfig(), device=self._device)

        bundle_paths = sorted(
            glob.glob(os.path.join(model_dir, "model_bundle_cluster_*.json"))
        )
        loaded_from_bundle: set = set()
        for bpath in bundle_paths:
            got = _load_cluster_from_bundle(bpath, model_dir, trainer)
            if got is None:
                continue
            cid, asset = got
            self.cluster_assets[cid] = asset
            loaded_from_bundle.add(cid)

        clusters_by_id: Dict[int, dict] = {}
        pkl = os.path.join(cluster_dir, "final_clusters.pkl")
        if os.path.isfile(pkl):
            clist = load_pickle(pkl)
            for c in clist:
                clusters_by_id[int(c["cluster_id"])] = c

        for path in glob.glob(os.path.join(norm_dir, "norm_params_cluster_*.json")):
            m = _NORM_CID_RE.search(os.path.basename(path))
            if not m:
                continue
            cid = int(m.group(1))
            if cid in loaded_from_bundle:
                continue
            params = NormalizationParams.load(path)
            model_path = os.path.join(model_dir, f"lstm_cluster_{cid}.pt")
            if not os.path.isfile(model_path):
                self.logger.warning("缺少模型: %s", model_path)
                continue
            model, cfg = trainer.load_model(model_path)
            cinfo = clusters_by_id.get(cid, {})
            flon = cinfo.get("rep_traj_lon")
            flat = cinfo.get("rep_traj_lat")
            if flon is None or flat is None:
                self.logger.warning("聚类 %s 缺少特征轨迹数组", cid)
                continue
            asset = ClusterAsset(
                cluster_id=cid,
                feature_traj_lon=np.asarray(flon, dtype=np.float64),
                feature_traj_lat=np.asarray(flat, dtype=np.float64),
                T=int(cfg["T"]),
                H=int(cfg["H"]),
                norm_params=params,
                model=model,
                model_path=model_path,
                norm_params_path=path,
            )
            self.cluster_assets[cid] = asset

        if self.predict_cfg.unknown_threshold is not None:
            self._effective_unknown_threshold = float(self.predict_cfg.unknown_threshold)
        else:
            cal_path = os.path.join(cluster_dir, "match_unknown_threshold.json")
            if os.path.isfile(cal_path):
                try:
                    data = load_json(cal_path)
                    self._effective_unknown_threshold = float(
                        data.get(
                            "unknown_threshold_suggested",
                            self.predict_cfg.unknown_threshold_fallback,
                        )
                    )
                    self.logger.info(
                        "未知类别阈值(离线标定): %s", self._effective_unknown_threshold
                    )
                except (OSError, TypeError, ValueError, KeyError) as e:
                    self.logger.warning(
                        "读取 match_unknown_threshold.json 失败: %s，使用 fallback", e
                    )
                    self._effective_unknown_threshold = float(
                        self.predict_cfg.unknown_threshold_fallback
                    )
            else:
                self._effective_unknown_threshold = float(
                    self.predict_cfg.unknown_threshold_fallback
                )
                self.logger.warning(
                    "未找到 match_unknown_threshold.json 且 predict.unknown_threshold 未设置，"
                    "使用 unknown_threshold_fallback=%s（方案 11.3 需按数据标定）",
                    self._effective_unknown_threshold,
                )

    def compute_match_distance(
        self,
        segment_lon: np.ndarray,
        segment_lat: np.ndarray,
        feature_lon: np.ndarray,
        feature_lat: np.ndarray,
        T: int,
    ) -> float:
        if len(segment_lon) != T or len(feature_lon) < T:
            return float("inf")
        best = float("inf")
        M = len(feature_lon)
        for j in range(0, M - T + 1):
            sub_lon = feature_lon[j : j + T]
            sub_lat = feature_lat[j : j + T]
            d = self.dtw.compute_distance_matrix_single_pair(
                segment_lon, segment_lat, sub_lon, sub_lat
            )
            if d < best:
                best = d
        return best

    def match_single_vessel(
        self,
        vessel_buffer: VesselBuffer,
        T_required: Optional[int] = None,
    ) -> Dict:
        best_id = None
        best_d = float("inf")
        second_d = float("inf")
        for cid, asset in self.cluster_assets.items():
            T = T_required if T_required is not None else asset.T
            seq = vessel_buffer.get_sequence(T)
            if seq.shape[0] < T:
                continue
            seg_lon = seq[:, 0]
            seg_lat = seq[:, 1]
            d = self.compute_match_distance(
                seg_lon,
                seg_lat,
                asset.feature_traj_lon,
                asset.feature_traj_lat,
                T,
            )
            if d < best_d:
                second_d = best_d
                best_d = d
                best_id = cid
            elif d < second_d:
                second_d = d
        ut = (
            self.predict_cfg.unknown_threshold
            if self.predict_cfg.unknown_threshold is not None
            else self._effective_unknown_threshold
        )
        is_unknown = (
            best_id is None
            or not np.isfinite(best_d)
            or (float(best_d) > float(ut))
        )
        if is_unknown:
            return {
                "cluster_id": -1,
                "match_distance": best_d,
                "model": None,
                "norm_params": None,
                "T": T_required or 0,
                "H": 0,
                "is_unknown": True,
            }
        asset = self.cluster_assets[best_id]
        return {
            "cluster_id": best_id,
            "match_distance": best_d,
            "model": asset.model,
            "norm_params": asset.norm_params,
            "T": asset.T,
            "H": asset.H,
            "is_unknown": False,
        }

    def _fork_allowed_cluster_ids(
        self, vessel_buffer: VesselBuffer
    ) -> Optional[Set[int]]:
        """
        None：未使用分叉口注册（Null）→ 保持原全局 Top-K 启发式。
        空集：非 Null 注册表下当前位置不在任何分叉口邻域 → 不触发 fork。
        非空：仅这些 cluster 可进入 fork 候选（方案 11.4 空间语义）。
        """
        reg = self.junction_registry
        if reg is None or isinstance(reg, NullForkJunctionRegistry):
            return None
        last = vessel_buffer.get_latest_point()
        if last is None:
            return set()
        _, lon, lat, _, _ = last
        jid = reg.junction_for_location(float(lon), float(lat))
        if jid is None:
            return set()
        return set(reg.clusters_at_junction(jid))

    def detect_fork_situation(
        self, vessel_buffer: VesselBuffer, top_k: Optional[int] = None
    ) -> Optional[List[Dict]]:
        allow = self._fork_allowed_cluster_ids(vessel_buffer)
        if allow is not None and len(allow) < 2:
            return None
        dists: List[Tuple[int, float]] = []
        for cid, asset in self.cluster_assets.items():
            if allow is not None and cid not in allow:
                continue
            T = asset.T
            seq = vessel_buffer.get_sequence(T)
            if seq.shape[0] < T:
                continue
            d = self.compute_match_distance(
                seq[:, 0],
                seq[:, 1],
                asset.feature_traj_lon,
                asset.feature_traj_lat,
                T,
            )
            dists.append((cid, d))
        dists.sort(key=lambda x: x[1])
        if len(dists) < 2:
            return None
        eff_k = (
            int(top_k)
            if top_k is not None
            else int(self.predict_cfg.max_fork_branches)
        )
        eff_k = max(2, eff_k)
        top = dists[: min(eff_k, len(dists))]
        if top[1][1] <= 0:
            return None
        if (top[1][1] - top[0][1]) / top[1][1] > 0.15:
            return None
        out = []
        for cid, d in top:
            a = self.cluster_assets[cid]
            out.append(
                {
                    "cluster_id": cid,
                    "match_distance": d,
                    "model": a.model,
                    "norm_params": a.norm_params,
                    "T": a.T,
                    "H": a.H,
                }
            )
        return out

    def compute_fork_probabilities(
        self,
        mmsi: int,
        candidate_cluster_ids: List[int],
    ) -> Dict[int, float]:
        """P(C_k)=N_k/ΣN_j，N_k 来自 MmsiBranchStatsStore（离线 mmsi_cluster_counts.json 预填）。"""
        if self.branch_stats_store is not None:
            counts = self.branch_stats_store.counts_for_mmsi_clusters(
                mmsi, candidate_cluster_ids
            )
        else:
            counts = {k: 1 for k in candidate_cluster_ids}
        s = sum(counts.values())
        if s <= 0:
            n = len(candidate_cluster_ids)
            return {k: 1.0 / n for k in candidate_cluster_ids} if n else {}
        return {k: counts.get(k, 0) / s for k in candidate_cluster_ids}
