"""
RSTPM 工具函数模块
==================
包含所有步骤共用的底层工具函数：
- Haversine 距离计算
- 航向角度插值（周期性处理）
- 日志初始化
- 通用 I/O 辅助
"""

import math
import logging
import os
import pickle
import json
from typing import Tuple, Optional, List, Any
from datetime import datetime

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 距离计算
# ═══════════════════════════════════════════════════════════════════════════════

def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float,
                       R: float = 6371000.0) -> float:
    """
    Haversine 公式计算两个经纬度点之间的地理距离（米）。
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2) ** 2
    a = min(1.0, max(0.0, a))
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_distance_vectorized(lat1: np.ndarray, lon1: np.ndarray,
                                   lat2: np.ndarray, lon2: np.ndarray,
                                   R: float = 6371000.0) -> np.ndarray:
    """向量化 Haversine（点对或广播）。"""
    phi1 = np.radians(lat1.astype(np.float64))
    phi2 = np.radians(lat2.astype(np.float64))
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(
        dlambda / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def euclidean_distance_lonlat(lon1: float, lat1: float,
                               lon2: float, lat2: float) -> float:
    """经纬度欧氏距离（DBSCAN 近似）。"""
    return math.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 航向角度处理
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_angle(angle: float) -> float:
    """将角度规范化到 [0, 360) 范围内。"""
    x = angle % 360.0
    if x < 0:
        x += 360.0
    return x


def interpolate_cog(cog1: float, cog2: float, alpha: float) -> float:
    """航向周期性线性插值（方案步骤5.4）。"""
    delta = cog2 - cog1
    if delta > 180:
        delta -= 360
    if delta < -180:
        delta += 360
    out = cog1 + alpha * delta
    return normalize_angle(out)


# ═══════════════════════════════════════════════════════════════════════════════
# 隐含速度计算
# ═══════════════════════════════════════════════════════════════════════════════

def compute_implied_speed(lat1: float, lon1: float, t1: float,
                           lat2: float, lon2: float, t2: float,
                           R: float = 6371000.0) -> float:
    """隐含速度 m/s；Δt<=0 返回 inf。"""
    dt = t2 - t1
    if dt <= 0:
        return float("inf")
    d = haversine_distance(lat1, lon1, lat2, lon2, R=R)
    return d / dt


# ═══════════════════════════════════════════════════════════════════════════════
# 弯曲系数计算
# ═══════════════════════════════════════════════════════════════════════════════

def compute_curvature_coefficient(trajectory_lons: np.ndarray,
                                   trajectory_lats: np.ndarray) -> float:
    """C = L / L'，直线时 C≈1。"""
    lons = np.asarray(trajectory_lons, dtype=np.float64).ravel()
    lats = np.asarray(trajectory_lats, dtype=np.float64).ravel()
    n = len(lons)
    if n < 2:
        return 1.0
    L_seg = 0.0
    for i in range(n - 1):
        L_seg += haversine_distance(
            lats[i], lons[i], lats[i + 1], lons[i + 1]
        )
    L_prime = haversine_distance(lats[0], lons[0], lats[-1], lons[-1])
    if L_prime < 1e-6:
        return 1.0
    return max(1.0, L_seg / L_prime)


def _lookup_curvature_map(curvature: float, mapping: dict) -> int:
    """mapping 的 key 为 (lower, upper) 半开区间 [lo, hi)。"""
    items = sorted(mapping.items(), key=lambda kv: kv[0][0])
    for (lo, hi), val in items:
        if hi == float("inf"):
            if curvature >= lo:
                return int(val)
        elif lo <= curvature < hi:
            return int(val)
    if items:
        return int(items[-1][1])
    return 12


def get_T_from_curvature(curvature: float, mapping: dict) -> int:
    """根据弯曲系数从配置映射中查找时间步长 T"""
    return _lookup_curvature_map(curvature, mapping)


def get_H_from_curvature(curvature: float, mapping: dict) -> int:
    """根据弯曲系数从配置映射中查找隐藏层神经元数 H"""
    return _lookup_curvature_map(curvature, mapping)


# ═══════════════════════════════════════════════════════════════════════════════
# 日志
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logger(name: str, log_dir: str = "./outputs/logs",
                 level: int = logging.INFO) -> logging.Logger:
    """控制台 + 文件日志；避免重复添加 handler。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    ensure_dir(log_dir)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d}.log"),
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# 通用 I/O
# ═══════════════════════════════════════════════════════════════════════════════

def save_pickle(obj: Any, filepath: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(filepath)) or ".")
    with open(filepath, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: str) -> Any:
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, filepath: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(filepath)) or ".")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Any:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(dir_path: str) -> None:
    """确保目录存在，不存在则创建"""
    os.makedirs(dir_path, exist_ok=True)
