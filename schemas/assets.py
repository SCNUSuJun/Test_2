"""聚类资产、模型清单与在线输出等结构化契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ClusterAsset:
    """
    单聚类类别在线侧所需资源（步骤 11 输入）。
    norm_params / model 在 load_cluster_assets 填充前可为 None。
    """

    cluster_id: int
    feature_traj_lon: Optional[Any] = None  # np.ndarray
    feature_traj_lat: Optional[Any] = None  # np.ndarray
    T: int = 0
    H: int = 0
    norm_params: Optional[Any] = None
    model: Optional[Any] = None
    model_path: str = ""
    norm_params_path: str = ""
    rep_traj_path: str = ""
    feature_order: Tuple[str, ...] = ("LON", "LAT", "SOG", "COG")
    schema_version: str = "1"
    ship_group: Optional[str] = None
    direction_label: Optional[str] = None
    roi_version: Optional[str] = None
    filter_profile: Optional[str] = None


@dataclass
class ModelBundleDescriptor:
    """单类别离线产物清单（步骤 9 保存、步骤 11/12 加载）。"""

    cluster_id: int
    T: int
    H: int
    input_size: int = 4
    output_size: int = 4
    feature_order: Tuple[str, ...] = ("LON", "LAT", "SOG", "COG")
    weights_path: str = ""
    norm_params_path: str = ""
    rep_traj_path: str = ""
    schema_version: str = "1"
    ship_group: Optional[str] = None
    direction_label: Optional[str] = None
    roi_version: Optional[str] = None
    filter_profile: Optional[str] = None
    git_commit: Optional[str] = None


@dataclass
class NormalizationParamsJSON:
    """归一化参数落盘 JSON 的字段契约（与 step7 序列化对齐）。"""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    sog_min: float
    sog_max: float
    cog_min: float
    cog_max: float
    cluster_id: Optional[int] = None
    schema_version: str = "1"


@dataclass
class OnlinePredictionResult:
    """步骤 12 单船预测输出（与 TrajectoryPredictor.format_prediction_output 对齐）。"""

    mmsi: int
    cluster_id: Optional[int]
    points: List[Dict[str, float]]  # lon, lat, sog, cog, t（t 为 Unix 秒）
    prediction_count: int = 0
    is_fork: bool = False
    fork_probabilities: Optional[Dict[int, float]] = None
    # 多分叉：cluster_id -> 该分支未来点序列（元素同 points 行）
    branch_predictions: Optional[Dict[int, List[Dict[str, float]]]] = None
    # 方案 11.4 第四步：跨调用消歧后本航段内固定使用该簇模型
    fork_locked_by_disambiguation: bool = False
    schema_version: str = "1"
