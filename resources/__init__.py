"""外部资源与几何数据访问抽象（多分叉等）。"""

from .fork_resources import (
    ChannelGeometryStore,
    MmsiBranchStatsStore,
    ForkJunctionRegistry,
)
from .implementations import (
    InMemoryMmsiBranchStatsStore,
    NullChannelGeometryStore,
    NullForkJunctionRegistry,
    prune_branch_predictions_by_geometry,
)

__all__ = [
    "ChannelGeometryStore",
    "MmsiBranchStatsStore",
    "ForkJunctionRegistry",
    "InMemoryMmsiBranchStatsStore",
    "NullChannelGeometryStore",
    "NullForkJunctionRegistry",
    "prune_branch_predictions_by_geometry",
]
