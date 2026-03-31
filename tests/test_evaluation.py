"""model/evaluation.py 附录 B 指标单测。"""

import os
import sys

import numpy as np
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


def test_distance_error_zero():
    from model.evaluation import PredictionEvaluator

    ev = PredictionEvaluator()
    lon = np.array([114.0, 114.01])
    lat = np.array([30.0, 30.0])
    d = ev.compute_distance_error(lon, lat, lon, lat)
    assert np.allclose(d, 0.0, atol=1e-3)


def test_heading_error_wrap():
    from model.evaluation import PredictionEvaluator

    ev = PredictionEvaluator()
    p = np.array([350.0, 10.0])
    t = np.array([10.0, 350.0])
    e = ev.compute_heading_error(p, t)
    assert e[0] == pytest.approx(20.0, abs=0.01)
    assert e[1] == pytest.approx(20.0, abs=0.01)


def test_evaluate_single_ade():
    from model.evaluation import PredictionEvaluator

    ev = PredictionEvaluator()
    pred = [
        {"lon": 114.0, "lat": 30.0, "sog": 5.0, "cog": 90.0},
        {"lon": 114.001, "lat": 30.0, "sog": 5.0, "cog": 90.0},
    ]
    gt = [
        {"lon": 114.0, "lat": 30.0, "sog": 5.0, "cog": 90.0},
        {"lon": 114.001, "lat": 30.0, "sog": 5.0, "cog": 90.0},
    ]
    m = ev.evaluate_single_trajectory(pred, gt)
    assert m["ade"] == pytest.approx(0.0, abs=1.0)
    assert m["mde"] == pytest.approx(0.0, abs=1.0)


def test_compute_accuracy_matches_formula():
    from model.evaluation import PredictionEvaluator

    ev = PredictionEvaluator()
    p = np.array([[0.5, 0.5, 0.5, 0.5]])
    t = np.array([[0.5, 0.5, 0.5, 0.5]])
    assert ev.compute_accuracy(p, t) == pytest.approx(1.0)


def test_length_mismatch_raises():
    from model.evaluation import PredictionEvaluator

    ev = PredictionEvaluator()
    with pytest.raises(ValueError):
        ev.compute_distance_error(
            np.array([1.0]), np.array([2.0, 3.0]), np.array([1.0]), np.array([2.0])
        )
