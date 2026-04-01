"""
方案 11.4 第四步：多航路分流下，随新 AIS 观测累积比较各分支上一时刻首步预测与当前位置偏差，锁定航路。

实现为工程可运行版本：每次在产生 fork 分支预测后缓存各分支首步预测点；下一
次满足预测条件时，用当前缓冲最新等间隔点与上一缓存的首步预测比 Haversine 距离，
累积误差；达到最少观测次数且最优分支相对次优满足比例阈值时锁定 cluster_id，
直至当前缓冲 segment 重置（超时清空）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils import haversine_distance


@dataclass
class ForkDisambiguationState:
    segment_id: int
    candidate_ids: Tuple[int, ...]
    last_branch_predictions: Optional[Dict[int, List[dict]]] = None
    cum_distance_m: Dict[int, float] = field(default_factory=dict)
    n_updates: int = 0
    locked_cluster_id: Optional[int] = None


def _first_pred_lon_lat(preds: Optional[List[dict]]) -> Optional[Tuple[float, float]]:
    if not preds:
        return None
    p0 = preds[0]
    lon = p0.get("lon")
    lat = p0.get("lat")
    if lon is None or lat is None:
        return None
    return float(lon), float(lat)


def update_fork_state_with_observation(
    state: ForkDisambiguationState,
    obs_lon: float,
    obs_lat: float,
    earth_radius: float,
) -> None:
    """用当前观测更新各候选分支的累积距离（相对上一时刻缓存的首步预测）。"""
    if state.locked_cluster_id is not None:
        return
    last = state.last_branch_predictions
    if not last:
        return
    for cid in state.candidate_ids:
        preds = last.get(cid)
        xy = _first_pred_lon_lat(preds)
        if xy is None:
            continue
        plon, plat = xy
        d = haversine_distance(obs_lat, obs_lon, plat, plon, R=earth_radius)
        state.cum_distance_m[cid] = state.cum_distance_m.get(cid, 0.0) + float(d)
    state.n_updates += 1


def try_lock_fork_state(
    state: ForkDisambiguationState,
    min_observations: int,
    error_ratio: float,
) -> Optional[int]:
    """
    若累积观测足够且最优分支总距离 * error_ratio < 次优总距离，则锁定最优。

    Returns
    -------
    cluster_id or None
    """
    if state.locked_cluster_id is not None:
        return state.locked_cluster_id
    if state.n_updates < int(min_observations):
        return None
    if len(state.candidate_ids) < 2:
        return None
    dists = []
    for cid in state.candidate_ids:
        dists.append((cid, state.cum_distance_m.get(cid, float("inf"))))
    dists.sort(key=lambda x: x[1])
    best_id, best_d = dists[0]
    _, second_d = dists[1]
    if not (best_d < float("inf") and second_d < float("inf")):
        return None
    if best_d * float(error_ratio) < second_d:
        state.locked_cluster_id = int(best_id)
        return state.locked_cluster_id
    return None


def candidates_compatible(
    prev: Tuple[int, ...], new_cands: List[int]
) -> bool:
    """候选集合一致时才延续累积（顺序无关）。"""
    return tuple(sorted(prev)) == tuple(sorted(new_cands))
