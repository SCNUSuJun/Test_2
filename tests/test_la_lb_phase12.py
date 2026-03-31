"""LA/LB 阶段一+二：Marine Cadastre 语义、域过滤、跨日切分、停泊标注（清单 47）。"""

import os
import sys

import pandas as pd
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


def test_step2_cog360_heading511_sentinels_keep_row():
    from config import FilterConfig, PortCleaningConfig
    from data_processing import AnomalyFilter

    f = FilterConfig(
        lon_min=114.0,
        lon_max=115.0,
        lat_min=29.0,
        lat_max=31.0,
        sog_max=25.0,
        allow_missing_cog_rows=True,
    )
    pc = PortCleaningConfig()
    df = pd.DataFrame(
        {
            "MMSI": [123456789],
            "Timestamp": [pd.to_datetime("2023-01-01T12:00:00")],
            "LON": [114.5],
            "LAT": [30.0],
            "SOG": [5.0],
            "COG": [360.0],
            "Heading": [511.0],
        }
    )
    af = AnomalyFilter(f, port_cleaning=pc)
    out = af.run(df)
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["COG"])
    assert pd.isna(out.iloc[0]["Heading"])


def test_step1_cheap_vessel_and_bbox_filter():
    from config import DataLoadConfig, DomainFocusConfig, PathConfig, PortCleaningConfig
    from data_processing import AISDataLoader

    path = PathConfig()
    load = DataLoadConfig()
    dom = DomainFocusConfig(enabled=True, vessel_type_keep=tuple(range(70, 80)))
    dom.roi_bbox = (-120.0, 33.0, -117.0, 35.0)
    loader = AISDataLoader(path, load, domain_config=dom, port_cleaning=PortCleaningConfig())
    df = pd.DataFrame(
        {
            "MMSI": [1, 2],
            "BaseDateTime": ["2023-01-01T00:00:00", "2023-01-01T00:01:00"],
            "LAT": [33.5, 33.6],
            "LON": [-118.2, -118.3],
            "SOG": [8.0, 9.0],
            "COG": [10.0, 20.0],
            "VesselType": [75, 90],
        }
    )
    df2 = loader._cheap_domain_filter(df)
    assert len(df2) == 1
    assert int(df2.iloc[0]["VesselType"]) == 75


def test_domain_spatial_filter_polygon(tmp_path):
    from config import DomainFocusConfig
    from data_processing import DomainSpatialFilter

    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
                },
            }
        ],
    }
    p = tmp_path / "roi.geojson"
    import json

    p.write_text(json.dumps(gj), encoding="utf-8")
    cfg = DomainFocusConfig(enabled=True, roi_polygon_path=str(p))
    df = pd.DataFrame(
        {
            "LON": [1.0, 3.0],
            "LAT": [1.0, 1.0],
            "MMSI": [1, 2],
            "Timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "SOG": [1.0, 1.0],
            "COG": [1.0, 1.0],
        }
    )
    out = DomainSpatialFilter(cfg).run(df)
    assert len(out) == 1


def test_cross_day_trajectory_not_split_by_calendar():
    from config import get_default_config
    from data_processing import AnomalyFilter, TrajectorySplitter

    cfg = get_default_config()
    cfg.filter.lon_min = 114.0
    cfg.filter.lon_max = 115.0
    cfg.filter.lat_min = 29.0
    cfg.filter.lat_max = 31.0
    cfg.traj_split.min_traj_length = 2
    rows = []
    t0 = pd.Timestamp("2023-06-01 23:50:00")
    for k in range(25):
        rows.append(
            dict(
                MMSI=123456789,
                Timestamp=t0 + pd.Timedelta(minutes=k * 2),
                LON=114.0 + k * 0.0001,
                LAT=30.0,
                SOG=5.0,
                COG=90.0,
            )
        )
    df = pd.DataFrame(rows)
    af = AnomalyFilter(cfg.filter, port_cleaning=cfg.port_cleaning)
    clean = af.run(df)
    sp = TrajectorySplitter(cfg.traj_split, domain_focus=cfg.domain_focus)
    trajs = sp.run(clean)
    assert len(trajs) == 1


def test_berth_label_only_preserves_rows_and_columns(tmp_path):
    from config import BerthConfig
    from data_processing import BerthDetector
    import json

    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }
    tp = tmp_path / "t.geojson"
    tp.write_text(json.dumps(gj), encoding="utf-8")
    bc = BerthConfig(
        mode="label_only",
        terminal_polygon_paths=[str(tp)],
        label_terminal_sog_max=10.0,
    )
    df = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": pd.to_datetime(["2023-01-01"]),
            "LON": [0.5],
            "LAT": [0.5],
            "SOG": [1.0],
            "COG": [10.0],
        }
    )
    det = BerthDetector(bc)
    out = det.run(df)
    assert len(out) == len(df)
    assert "is_terminal_dwell" in out.columns
    assert bool(out.iloc[0]["is_terminal_dwell"])
