"""
步骤10：在线阶段——接收实时AIS数据并进行预处理
================================================

缓冲区语义与离线 `data_processing/step5_resample.py` 对齐：仅保留落在等间隔时间网格
`t0 + k * resample_interval`（k 为整数）上的状态点。原始 AIS 观测只作为线段端点参与
线性/圆周插值生成网格点，**不**再以非网格时间戳入缓冲（首条报文时间作为轨迹起点 t0，
与离线首点时间一致）。
原始 AIS 按时间追加至 ``VesselBuffer.ais_history``，网格时刻为上一网格时刻加 ``resample_interval``，
状态由该链上 ``searchsorted`` 分段线性得到，与 ``step5_resample.resample_single_trajectory`` 一致。
"""

import numpy as np
from typing import Dict, Optional, Tuple, List

from config import FilterConfig, ResampleConfig, PredictConfig
from utils import (
    haversine_distance,
    interpolate_cog,
    setup_logger,
)


class VesselBuffer:
    """单船滑动缓冲区：等间隔点 (timestamp_unix, lon, lat, sog, cog)"""

    def __init__(self, mmsi: int, max_length: int):
        self.mmsi = mmsi
        self.max_length = max_length
        self.data: List[Tuple[float, float, float, float, float]] = []
        # 自当前航段起按时间排序的原始 AIS，用于与离线 step5 searchsorted 分段线性一致（清单 43）
        self.ais_history: List[Tuple[float, float, float, float, float]] = []

    def is_ready(self, required_length: int) -> bool:
        return len(self.data) >= required_length

    def get_latest_point(self) -> Optional[Tuple[float, float, float, float, float]]:
        if not self.data:
            return None
        return self.data[-1]

    def clear(self) -> None:
        self.data.clear()
        self.ais_history.clear()

    def append(self, point: Tuple[float, float, float, float, float]) -> None:
        self.data.append(point)
        while len(self.data) > self.max_length:
            self.data.pop(0)

    def set_max_length(self, max_length: int) -> None:
        """与离线资产 T 对齐时扩容缓冲（步骤10）；超长尾部丢弃。"""
        self.max_length = int(max_length)
        while len(self.data) > self.max_length:
            self.data.pop(0)

    def get_sequence(self, length: int) -> np.ndarray:
        if length > len(self.data):
            length = len(self.data)
        seg = self.data[-length:]
        return np.array([[p[1], p[2], p[3], p[4]] for p in seg], dtype=np.float64)


class RealtimePreprocessor:
    """实时 AIS 预处理与缓冲区管理。"""

    def __init__(
        self,
        filter_config: FilterConfig,
        resample_config: ResampleConfig,
        predict_config: PredictConfig,
        max_T: int = 14,
    ):
        self.filter_cfg = filter_config
        self.resample_cfg = resample_config
        self.predict_cfg = predict_config
        self.max_T = max_T
        self.logger = setup_logger("RealtimePreprocessor")
        self.buffers: Dict[int, VesselBuffer] = {}

    def validate_message(
        self,
        mmsi: int,
        timestamp: float,
        lon: float,
        lat: float,
        sog: float,
        cog: float,
    ) -> bool:
        c = self.filter_cfg
        if not (c.lon_min <= lon <= c.lon_max and c.lat_min <= lat <= c.lat_max):
            return False
        if sog != sog or sog < c.sog_min or sog > c.sog_max:
            return False
        if cog != cog or cog < c.cog_min or cog >= c.cog_max:
            return False
        return True

    def check_jump_distance(
        self,
        buffer: VesselBuffer,
        new_lat: float,
        new_lon: float,
        new_timestamp: float,
    ) -> bool:
        last = buffer.get_latest_point()
        if last is None:
            return True
        c = self.filter_cfg
        v_thresh = c.sog_max * c.knot_to_ms * c.jump_speed_factor
        dt = new_timestamp - last[0]
        if dt <= 0:
            return False
        d = haversine_distance(last[2], last[1], new_lat, new_lon, R=c.earth_radius)
        return (d / dt) <= v_thresh

    @staticmethod
    def _state_at_time_unix(
        hist: List[Tuple[float, float, float, float, float]], tv: float
    ) -> Tuple[float, float, float, float]:
        """与 data_processing/step5_resample.py 中 searchsorted 括号一致的分段线性 + COG 圆周。"""
        ts_orig = np.array([h[0] for h in hist], dtype=np.float64)
        if len(ts_orig) < 2:
            h0 = hist[0]
            return h0[1], h0[2], h0[3], h0[4]
        lon_o = np.array([h[1] for h in hist], dtype=np.float64)
        lat_o = np.array([h[2] for h in hist], dtype=np.float64)
        sog_o = np.array([h[3] for h in hist], dtype=np.float64)
        cog_o = np.array([h[4] for h in hist], dtype=np.float64)
        tv = float(tv)
        if tv <= ts_orig[0] + 1e-9:
            return float(lon_o[0]), float(lat_o[0]), float(sog_o[0]), float(cog_o[0])
        if tv >= ts_orig[-1] - 1e-9:
            return float(lon_o[-1]), float(lat_o[-1]), float(sog_o[-1]), float(cog_o[-1])
        idx = int(np.searchsorted(ts_orig, tv, side="right"))
        if idx <= 0:
            i0, i1 = 0, 1
        elif idx >= len(ts_orig):
            i0, i1 = len(ts_orig) - 2, len(ts_orig) - 1
        else:
            i1 = idx
            i0 = idx - 1
        t_a, t_b = float(ts_orig[i0]), float(ts_orig[i1])
        if t_b <= t_a:
            alpha = 0.0
        else:
            alpha = (tv - t_a) / (t_b - t_a)
        lon_i = lon_o[i0] + alpha * (lon_o[i1] - lon_o[i0])
        lat_i = lat_o[i0] + alpha * (lat_o[i1] - lat_o[i0])
        sog_i = sog_o[i0] + alpha * (sog_o[i1] - sog_o[i0])
        cog_i = interpolate_cog(float(cog_o[i0]), float(cog_o[i1]), alpha)
        return float(lon_i), float(lat_i), float(sog_i), float(cog_i)

    def interpolate_and_fill(
        self,
        buffer: VesselBuffer,
        new_point: Tuple[float, float, float, float, float],
    ) -> None:
        ts_n, lon_n, lat_n, sog_n, cog_n = new_point
        last = buffer.get_latest_point()
        dt_grid = self.resample_cfg.resample_interval
        if last is None:
            buffer.append(new_point)
            buffer.ais_history = [new_point]
            return
        t_last_buf, _, _, _, _ = last
        if ts_n <= t_last_buf:
            return
        span = ts_n - t_last_buf
        if span > self.predict_cfg.buffer_timeout:
            buffer.clear()
            buffer.append(new_point)
            buffer.ais_history = [new_point]
            return
        if not buffer.ais_history:
            buffer.ais_history = [buffer.data[0]]
        lh = buffer.ais_history[-1][0]
        if abs(ts_n - lh) < 1e-6:
            buffer.ais_history[-1] = new_point
        elif ts_n > lh:
            buffer.ais_history.append(new_point)
        else:
            return
        last_grid_t = buffer.data[-1][0]
        t_end = buffer.ais_history[-1][0]
        while last_grid_t + dt_grid <= t_end + 1e-6:
            tv = last_grid_t + dt_grid
            lon_i, lat_i, sog_i, cog_i = self._state_at_time_unix(buffer.ais_history, tv)
            buffer.append((tv, lon_i, lat_i, sog_i, cog_i))
            last_grid_t = tv

    def process_message(
        self,
        mmsi: int,
        timestamp: float,
        lon: float,
        lat: float,
        sog: float,
        cog: float,
    ) -> Optional[int]:
        if not self.validate_message(mmsi, timestamp, lon, lat, sog, cog):
            return None
        ts = float(timestamp)
        buf = self.buffers.setdefault(mmsi, VesselBuffer(mmsi, self.max_T))
        new_pt = (ts, lon, lat, sog, cog)
        last = buf.get_latest_point()
        # 方案步骤 2.7 / 10：重复点以「最近原始 AIS 时间」为参照，避免缓冲末点为网格时间时误拒真报文
        dup_th = float(self.filter_cfg.duplicate_time_threshold)
        dup_ref = buf.ais_history[-1][0] if buf.ais_history else None
        if dup_ref is not None:
            if abs(ts - dup_ref) < 1e-6:
                buf.ais_history[-1] = new_pt
                self.interpolate_and_fill(buf, new_pt)
                return mmsi if buf.is_ready(self.max_T) else None
            if 0 < ts - dup_ref < dup_th:
                return None
        elif last is not None and (ts - last[0]) < dup_th:
            return None
        if last is not None and ts - last[0] > self.predict_cfg.buffer_timeout:
            buf.clear()
        if last is not None and not self.check_jump_distance(buf, lat, lon, ts):
            return None
        self.interpolate_and_fill(buf, new_pt)
        if buf.is_ready(self.max_T):
            return mmsi
        return None

    def get_vessel_buffer(self, mmsi: int) -> Optional[VesselBuffer]:
        return self.buffers.get(mmsi)

    def get_active_vessels(self) -> List[int]:
        return list(self.buffers.keys())

    def set_buffer_capacity(self, max_T: int) -> None:
        """更新所有在航船舶缓冲容量（加载模型后可能与各簇 T 对齐）。"""
        self.max_T = int(max_T)
        for buf in self.buffers.values():
            buf.set_max_length(self.max_T)
