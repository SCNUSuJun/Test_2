"""方案 11.4 第四步：分叉消歧状态与资源加载烟测。"""

import os

import pytest

from online.fork_disambiguation import (
    ForkDisambiguationState,
    try_lock_fork_state,
    update_fork_state_with_observation,
)
from resources import load_channel_geometry_store, load_fork_junction_registry


def test_fork_lock_after_two_updates():
    st = ForkDisambiguationState(
        segment_id=0,
        candidate_ids=(0, 1),
        last_branch_predictions={
            0: [{"lon": 0.0, "lat": 0.0, "timestamp": 1.0}],
            1: [{"lon": 10.0, "lat": 10.0, "timestamp": 1.0}],
        },
    )
    update_fork_state_with_observation(st, 0.001, 0.001, 6371000.0)
    update_fork_state_with_observation(st, 0.001, 0.001, 6371000.0)
    locked = try_lock_fork_state(st, min_observations=2, error_ratio=1.35)
    assert locked == 0
    assert st.locked_cluster_id == 0


def test_load_channel_geojson_default():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "resources", "channel_centerlines.geojson")
    store = load_channel_geometry_store(path if os.path.isfile(path) else None)
    if os.path.isfile(path):
        ids = store.list_channel_ids_near(-118.35, 33.52, radius_m=500_000.0)
        assert len(ids) >= 1


def test_load_empty_fork_registry_is_null():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "resources", "fork_junctions.json")
    reg = load_fork_junction_registry(path if os.path.isfile(path) else None)
    from resources import NullForkJunctionRegistry

    assert isinstance(reg, NullForkJunctionRegistry)
