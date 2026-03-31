"""
离线估计在线类别匹配的「未知阈值」上界参考（方案 11.3）。

用各类别特征轨迹之间、固定长度 T 的滑窗改进 DTW 交叉匹配距离分布，
取高分位数作为 suggested 阈值：best_d 大于该值时倾向判为未知（需与业务联调）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from clustering.step6c_dtw_refine import ImprovedDTW


def _sliding_match_dtw(
    segment_lon: np.ndarray,
    segment_lat: np.ndarray,
    feature_lon: np.ndarray,
    feature_lat: np.ndarray,
    T: int,
    dtw: ImprovedDTW,
) -> float:
    if len(segment_lon) != T or len(feature_lon) < T:
        return float("inf")
    best = float("inf")
    M = len(feature_lon)
    for j in range(0, M - T + 1):
        sub_lon = feature_lon[j : j + T]
        sub_lat = feature_lat[j : j + T]
        d = dtw.compute_distance_matrix_single_pair(
            segment_lon, segment_lat, sub_lon, sub_lat
        )
        if d < best:
            best = d
    return float(best)


def estimate_unknown_threshold_from_clusters(
    final_clusters: List[Dict[str, Any]],
    calibration_T: int = 12,
    percentile: float = 75.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    从最终聚类特征轨迹估计 unknown_threshold_suggested。

    Returns
    -------
    suggested : float
    meta : dict
        写入 JSON 的辅助字段（schema_version、样本数等）。
    """
    dtw = ImprovedDTW()
    feats: List[Tuple[np.ndarray, np.ndarray]] = []
    for c in final_clusters:
        rl = c.get("rep_traj_lon")
        ra = c.get("rep_traj_lat")
        if rl is None or ra is None:
            continue
        lon = np.asarray(rl, dtype=np.float64)
        lat = np.asarray(ra, dtype=np.float64)
        if len(lon) < 2:
            continue
        feats.append((lon, lat))
    K = len(feats)
    if K < 2:
        fb = 0.15
        return fb, {
            "schema_version": "1",
            "unknown_threshold_suggested": fb,
            "reason": "insufficient_clusters",
            "n_clusters": K,
            "calibration_T": int(calibration_T),
        }

    T = int(calibration_T)
    min_len = min(len(f[0]) for f in feats)
    T = max(2, min(T, min_len))
    cross_vals: List[float] = []
    for i in range(K):
        lon_i, lat_i = feats[i]
        if len(lon_i) < T:
            continue
        seg_lon = lon_i[:T]
        seg_lat = lat_i[:T]
        for j in range(K):
            if i == j:
                continue
            lon_j, lat_j = feats[j]
            d = _sliding_match_dtw(seg_lon, seg_lat, lon_j, lat_j, T, dtw)
            if np.isfinite(d):
                cross_vals.append(d)

    if not cross_vals:
        fb = 0.15
        return fb, {
            "schema_version": "1",
            "unknown_threshold_suggested": fb,
            "reason": "no_cross_values",
            "n_clusters": K,
            "calibration_T": T,
        }

    suggested = float(np.percentile(cross_vals, percentile))
    suggested = max(suggested, 1e-8)
    meta = {
        "schema_version": "1",
        "unknown_threshold_suggested": suggested,
        "calibration_T": T,
        "percentile": float(percentile),
        "n_cross_samples": len(cross_vals),
        "n_clusters": K,
    }
    return suggested, meta
