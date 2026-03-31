"""
核心工具函数与关键步骤的单元测试
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestHaversine:
    """Haversine 距离计算测试"""
    
    def test_same_point(self):
        """同一点距离为0"""
        from utils import haversine_distance
        d = haversine_distance(30.0, 114.0, 30.0, 114.0)
        assert d == pytest.approx(0.0, abs=1e-6)
    
    def test_known_distance(self):
        """已知距离验证（北京到上海约1068km）"""
        from utils import haversine_distance
        d = haversine_distance(39.9042, 116.4074, 31.2304, 121.4737)
        assert 1060000 < d < 1080000  # 约1068km
    
    def test_vectorized(self):
        """向量化版本与逐点版本结果一致"""
        from utils import haversine_distance, haversine_distance_vectorized

        lat1 = np.array([30.0, 31.0])
        lon1 = np.array([114.0, 115.0])
        lat2 = np.array([30.1, 31.1])
        lon2 = np.array([114.1, 115.1])
        dv = haversine_distance_vectorized(lat1, lon1, lat2, lon2)
        for i in range(2):
            d0 = haversine_distance(lat1[i], lon1[i], lat2[i], lon2[i])
            assert dv[i] == pytest.approx(d0, rel=1e-5)


class TestAngleInterpolation:
    """航向角度插值测试"""
    
    def test_normal_interpolation(self):
        """普通角度插值"""
        from utils import interpolate_cog
        result = interpolate_cog(90.0, 180.0, 0.5)
        assert result == pytest.approx(135.0, abs=0.01)
    
    def test_wrap_around_interpolation(self):
        """跨越 0°/360° 的插值（350° → 10° 中间应为 0°）"""
        from utils import interpolate_cog
        result = interpolate_cog(350.0, 10.0, 0.5)
        assert result == pytest.approx(0.0, abs=0.01)
    
    def test_reverse_wrap(self):
        """反方向跨越（10° → 350° 中间应为 0°）"""
        from utils import interpolate_cog
        result = interpolate_cog(10.0, 350.0, 0.5)
        assert result == pytest.approx(0.0, abs=0.01)


class TestCurvatureCoefficient:
    """弯曲系数计算测试"""
    
    def test_straight_line(self):
        """直线轨迹弯曲系数应为1.0"""
        from utils import compute_curvature_coefficient
        lons = np.array([114.0, 114.1, 114.2, 114.3])
        lats = np.array([30.0, 30.0, 30.0, 30.0])
        C = compute_curvature_coefficient(lons, lats)
        assert C == pytest.approx(1.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# 改进DTW测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestImprovedDTW:
    """改进DTW距离计算测试"""
    
    def test_identical_trajectories(self):
        """相同轨迹的DTW距离为0"""
        from clustering.step6c_dtw_refine import ImprovedDTW
        lon = np.array([114.0, 114.1, 114.2])
        lat = np.array([30.0, 30.1, 30.2])
        d = ImprovedDTW.compute_distance_matrix_single_pair(lon, lat, lon, lat)
        assert d == pytest.approx(0.0, abs=1e-6)
    
    def test_different_trajectories(self):
        """不同轨迹的DTW距离大于0"""
        from clustering.step6c_dtw_refine import ImprovedDTW
        lon_a = np.array([114.0, 114.1, 114.2])
        lat_a = np.array([30.0, 30.1, 30.2])
        lon_b = np.array([115.0, 115.1, 115.2])
        lat_b = np.array([31.0, 31.1, 31.2])
        d = ImprovedDTW.compute_distance_matrix_single_pair(lon_a, lat_a, lon_b, lat_b)
        assert d > 0


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM模型测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestLSTMModel:
    """LSTM模型结构测试"""
    
    def test_model_output_shape(self):
        """模型输出形状正确"""
        import torch
        from model.step9_lstm_train import LSTMPredictor
        model = LSTMPredictor(input_size=4, hidden_size=128)
        x = torch.randn(32, 12, 4)  # batch=32, T=12, features=4
        y = model(x)
        assert y.shape == (32, 4)
    
    def test_model_different_T(self):
        """不同时间步长T的模型都能正常前向传播"""
        import torch
        from model.step9_lstm_train import LSTMPredictor
        model = LSTMPredictor(input_size=4, hidden_size=88)
        for T in [8, 10, 12, 14]:
            x = torch.randn(16, T, 4)
            y = model(x)
            assert y.shape == (16, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# 归一化测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestOnlinePredictionSchema:
    """步骤 12 输出与 schemas.OnlinePredictionResult 对齐。"""

    def test_format_prediction_output_points(self):
        from online.step12_prediction import TrajectoryPredictor
        from config import get_default_config

        cfg = get_default_config()
        pred = TrajectoryPredictor(cfg.predict, cfg.resample, device="cpu")
        raw = [
            {
                "step": 1,
                "timestamp": 1000.0,
                "lon": 114.0,
                "lat": 30.0,
                "sog": 5.0,
                "cog": 90.0,
            }
        ]
        out = pred.format_prediction_output(123, 0, raw, is_fork=False)
        assert "points" in out and "prediction_count" in out
        assert out["points"][0]["t"] == 1000.0
        assert out["schema_version"] == "1"

    def test_format_prediction_requires_timestamp(self):
        from online.step12_prediction import TrajectoryPredictor
        from config import get_default_config
        import pytest

        cfg = get_default_config()
        pred = TrajectoryPredictor(cfg.predict, cfg.resample, device="cpu")
        bad = [{"lon": 114.0, "lat": 30.0, "sog": 5.0, "cog": 90.0}]
        with pytest.raises(ValueError, match="timestamp"):
            pred.format_prediction_output(1, 0, bad, is_fork=False)


class TestNormalization:
    """归一化与反归一化测试"""
    
    def test_round_trip(self):
        """归一化后反归一化应还原原始值"""
        from model.step7_normalization import NormalizationParams, DataNormalizer
        params = NormalizationParams(
            lon_min=114.0, lon_max=115.0,
            lat_min=30.0, lat_max=31.0,
            sog_min=0.0, sog_max=20.0,
            cog_min=0.0, cog_max=360.0,
        )
        # 测试某个点
        lon, lat, sog, cog = 114.5, 30.5, 10.0, 180.0
        # 归一化
        lon_n = (lon - params.lon_min) / (params.lon_max - params.lon_min)
        lat_n = (lat - params.lat_min) / (params.lat_max - params.lat_min)
        # 反归一化
        lon_r = lon_n * (params.lon_max - params.lon_min) + params.lon_min
        lat_r = lat_n * (params.lat_max - params.lat_min) + params.lat_min
        assert lon_r == pytest.approx(lon, abs=1e-10)
        assert lat_r == pytest.approx(lat, abs=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
