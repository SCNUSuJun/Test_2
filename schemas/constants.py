"""全链路冻结列名与 schema 版本（仅契约，运行时校验可选用）。"""

from typing import Dict, Final, Tuple

# Marine Cadastre → 内部时间列（清单 6）
MARINE_CADASTRE_TIMESTAMP_FIELD: Final[str] = "BaseDateTime"
INTERNAL_TIMESTAMP_FIELD: Final[str] = "Timestamp"

# 常用列名别名（大小写以数据字典为准时可在此扩展）
FIELD_ALIASES: Final[Dict[str, str]] = {
    "BaseDateTime": "Timestamp",
    "basdatetime": "Timestamp",
}

# 步骤 1–5 主轨迹点最小列（模型仍仅用 4 维特征）
TRAJECTORY_POINT_COLUMNS: Final[Tuple[str, ...]] = (
    "MMSI",
    "Timestamp",
    "LON",
    "LAT",
    "SOG",
    "COG",
)

TRAJECTORY_WITH_ID_COLUMNS: Final[Tuple[str, ...]] = (
    "MMSI",
    "TrajID",
    "Timestamp",
    "LON",
    "LAT",
    "SOG",
    "COG",
)

FEATURE_ORDER_LONLAT: Final[Tuple[str, ...]] = ("LON", "LAT", "SOG", "COG")

SCHEMA_VERSION: Final[str] = "rstpm-schema-v1"
