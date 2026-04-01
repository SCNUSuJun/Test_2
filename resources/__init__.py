"""外部资源与几何数据访问抽象（多分叉等）。"""

from .fork_resources import (
    ChannelGeometryStore,
    MmsiBranchStatsStore,
    ForkJunctionRegistry,
)
from .implementations import (
    GeoJsonLineStringChannelStore,
    InMemoryMmsiBranchStatsStore,
    JsonForkJunctionRegistry,
    NullChannelGeometryStore,
    NullForkJunctionRegistry,
    load_channel_geometry_store,
    load_fork_junction_registry,
    prune_branch_predictions_by_geometry,
)

__all__ = [
    "ChannelGeometryStore",
    "MmsiBranchStatsStore",
    "ForkJunctionRegistry",
    "GeoJsonLineStringChannelStore",
    "JsonForkJunctionRegistry",
    "InMemoryMmsiBranchStatsStore",
    "NullChannelGeometryStore",
    "NullForkJunctionRegistry",
    "load_channel_geometry_store",
    "load_fork_junction_registry",
    "prune_branch_predictions_by_geometry",
]
