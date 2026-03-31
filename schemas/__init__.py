"""跨模块数据契约（列名、资产结构、在线输出）。"""

from .constants import (
    TRAJECTORY_POINT_COLUMNS,
    TRAJECTORY_WITH_ID_COLUMNS,
    FEATURE_ORDER_LONLAT,
    SCHEMA_VERSION,
)
from .assets import (
    ClusterAsset,
    ModelBundleDescriptor,
    NormalizationParamsJSON,
    OnlinePredictionResult,
)

__all__ = [
    "TRAJECTORY_POINT_COLUMNS",
    "TRAJECTORY_WITH_ID_COLUMNS",
    "FEATURE_ORDER_LONLAT",
    "SCHEMA_VERSION",
    "ClusterAsset",
    "ModelBundleDescriptor",
    "NormalizationParamsJSON",
    "OnlinePredictionResult",
]
