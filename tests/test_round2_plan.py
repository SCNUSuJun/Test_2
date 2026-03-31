"""
第二轮精准修复计划验收：步骤10 等间隔缓冲、步骤8 按年划分（清单 36/43）。
"""

import os
import sys

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


def test_step10_buffer_only_grid_points_120s():
    from config import FilterConfig, ResampleConfig, PredictConfig
    from online.step10_realtime_preprocess import RealtimePreprocessor

    fc = FilterConfig(
        lon_min=-180.0,
        lon_max=180.0,
        lat_min=-90.0,
        lat_max=90.0,
        sog_max=30.0,
    )
    rc = ResampleConfig(resample_interval=120.0)
    pc = PredictConfig(buffer_timeout=1.0e9)
    pre = RealtimePreprocessor(fc, rc, pc, max_T=10)
    mmsi = 123456789

    def feed(ts: float) -> None:
        pre.process_message(mmsi, ts, -118.3, 33.7, 5.0, 90.0)

    feed(0.0)
    buf = pre.get_vessel_buffer(mmsi)
    assert buf is not None
    assert len(buf.data) == 1
    assert abs(buf.data[0][0] - 0.0) < 1e-6

    feed(30.0)
    assert len(buf.data) == 1

    feed(150.0)
    assert len(buf.data) == 2
    assert abs(buf.data[1][0] - buf.data[0][0] - 120.0) < 1e-3


def test_step8_time_split_test_set_excludes_train_year_trajectories():
    from config import SplitConfig, TrainConfig
    from model.step8_sample_construction import SampleConstructor

    n = 32
    t0_23 = pd.Timestamp("2023-06-15 12:00:00")
    t0_24 = pd.Timestamp("2024-06-15 12:00:00")
    ts23 = [t0_23 + pd.Timedelta(seconds=i * 120) for i in range(n)]
    ts24 = [t0_24 + pd.Timedelta(seconds=i * 120) for i in range(n)]

    df23 = pd.DataFrame(
        {
            "Timestamp": ts23,
            "LON": np.linspace(-118.4, -118.3, n, dtype=np.float64),
            "LAT": np.ones(n, dtype=np.float64),
            "SOG": np.full(n, 8.0),
            "COG": np.full(n, 10.0),
        }
    )
    df24 = pd.DataFrame(
        {
            "Timestamp": ts24,
            "LON": np.linspace(-118.4, -118.3, n, dtype=np.float64),
            "LAT": np.full(n, 2.0, dtype=np.float64),
            "SOG": np.full(n, 8.0),
            "COG": np.full(n, 10.0),
        }
    )

    split = SplitConfig(
        train_years=(2023,),
        val_years=(),
        test_years=(2024,),
        time_based_split=True,
        split_by_voyage=True,
    )
    sc = SampleConstructor(TrainConfig(), split)
    T = 5
    train_ds, _val_ds, test_ds = sc.split_by_trajectory([df23, df24], T)
    assert len(train_ds) > 0
    assert len(test_ds) > 0
    # 标签第 2 列为 LAT；2024 轨迹 LAT=2.0，2023 为 1.0
    te_lat = test_ds.labels.numpy()[:, 1]
    assert np.all(te_lat > 1.5)
    tr_lat = train_ds.labels.numpy()[:, 1]
    assert np.all(tr_lat < 1.5)
