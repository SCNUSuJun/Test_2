"""最小离线链路：合成数据走步骤 2→5 与聚类 6A→6B。"""

import os
import sys

import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


def _synthetic_track(n: int = 25, dt_s: int = 100) -> pd.DataFrame:
    t0 = pd.Timestamp("2023-06-01 10:00:00")
    rows = []
    for i in range(n):
        rows.append(
            {
                "MMSI": 123456789,
                "Timestamp": t0 + pd.Timedelta(seconds=i * dt_s),
                "LON": 114.0 + i * 0.0001,
                "LAT": 30.0 + i * 0.00005,
                "SOG": 5.0,
                "COG": 90.0,
            }
        )
    return pd.DataFrame(rows)


def test_steps_2_to_5():
    from config import get_default_config
    from data_processing import AnomalyFilter, BerthDetector, TrajectorySplitter
    from data_processing.step5_resample import TrajectoryResampler

    cfg = get_default_config()
    df = _synthetic_track(30, dt_s=200)
    af = AnomalyFilter(cfg.filter)
    clean = af.run(df)
    assert len(clean) >= len(df) * 0.8
    bd = BerthDetector(cfg.berth)
    nb = bd.run(clean)
    sp = TrajectorySplitter(cfg.traj_split)
    trajs = sp.run(nb)
    assert len(trajs) >= 1
    rs = TrajectoryResampler(cfg.resample)
    out = rs.run(trajs)
    assert len(out) >= 1
    assert len(out[0]) >= 2


def test_cluster_6a_6b():
    from config import get_default_config
    from clustering.step6a_endpoints import EndpointExtractor
    from clustering.step6b_dbscan import StartEndClusterer

    cfg = get_default_config()
    trajs = []
    for k in range(8):
        base_lon = 114.0 + k * 0.002
        t0 = pd.Timestamp("2023-01-01")
        pts = []
        for i in range(22):
            pts.append(
                {
                    "MMSI": 100000000 + k,
                    "TrajID": k,
                    "Timestamp": t0 + pd.Timedelta(minutes=i * 2),
                    "LON": base_lon + i * 0.0001,
                    "LAT": 30.0 + i * 0.0001,
                    "SOG": 4.0,
                    "COG": 45.0,
                }
            )
        trajs.append(pd.DataFrame(pts))
    ex = EndpointExtractor()
    sp, ep, idx = ex.extract(trajs)
    assert len(idx) == 8
    cl = StartEndClusterer(cfg.cluster)
    groups = cl.run(sp, ep, idx)
    assert isinstance(groups, dict)
