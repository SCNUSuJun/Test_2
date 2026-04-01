"""
RSTPM 全局配置文件
==================
所有超参数、路径、阈值均集中定义于此，方便统一管理和调参。
参数来源：RSTPM_full_pipeline.md 附录A 关键参数速查表。

参数层级（框架约定，避免将「论文示例」与「项目定案」混为一谈）：
- paper_default：附录 A / 论文实验常用值，实现时可作起点，不等于业务强制值。
- dataset_validated：已针对当前 AIS 数据集确认的路径、列名、研究区边界等。
- needs_tuning：需离线实验或在线调优后再定（如 unknown_threshold、merge_threshold）。

LA/LB 领域块见 DomainFocusConfig / PortCleaningConfig；分层说明见 PARAM_LAYERS.md。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PathConfig:
    """数据与输出路径配置"""

    # --- 原始数据 ---
    # [dataset_validated] 必须按实际数据位置修改
    raw_data_dir: str = "/path/to/AIS_zips"  # 存放 AIS_YYYY_MM_DD.zip 的目录

    # --- LA/LB 子集产物（清单 9）---
    subsets_root: str = "./outputs/subsets"
    subsets_raw_daily_dir: str = "./outputs/subsets/raw_daily_filtered"
    subsets_metadata_dir: str = "./outputs/subsets/metadata"
    subsets_merged_monthly_dir: str = "./outputs/subsets/merged_monthly"
    subset_checkpoint_path: str = "./outputs/subsets/metadata/subset_checkpoint.json"

    # --- 中间数据输出 ---
    intermediate_dir: str = "./outputs/intermediate"  # 各步骤的中间产物
    # [paper_default + engineering] 大规模数据：按日/按 zip 落盘，禁止默认全量拼单表常驻内存
    step1_daily_output_dir: str = "./outputs/intermediate/step1_by_day"
    pipeline_checkpoint_path: str = "./outputs/intermediate/pipeline_checkpoint.json"
    cleaned_data_dir: str = "./outputs/intermediate/step2_cleaned"
    no_berth_data_dir: str = "./outputs/intermediate/step3_no_berth"
    trajectories_dir: str = "./outputs/intermediate/step4_trajectories"
    resampled_dir: str = "./outputs/intermediate/step5_resampled"

    # [engineering] 为 True 时在步骤 2/3/4 后落盘中间表与 stepN_complete.json（默认关闭，避免大规模 IO）
    persist_intermediate_tables: bool = False

    # --- 聚类结果 ---
    cluster_dir: str = "./outputs/clusters"

    # --- 归一化参数 ---
    normalization_dir: str = "./outputs/normalization"

    # --- 模型 ---
    model_dir: str = "./outputs/models"

    # --- 日志与图表 ---
    log_dir: str = "./outputs/logs"
    figure_dir: str = "./outputs/figures"


# ═══════════════════════════════════════════════════════════════════════════════
# LA/LB 领域与港口清洗（清单 4）
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DomainFocusConfig:
    """研究区、船型、子集输出；启用后 step1/step1b 走 LA/LB 管线。"""

    enabled: bool = False
    study_name: str = "LA_LB_merchant_AIS"
    years: List[int] = field(default_factory=lambda: [2023, 2024])
    vessel_type_keep: Tuple[int, ...] = tuple(range(70, 90))
    ship_groups: Dict[str, Tuple[int, ...]] = field(
        default_factory=lambda: {
            "cargo": tuple(range(70, 80)),
            "tanker": tuple(range(80, 90)),
        }
    )
    transceiver_class_keep: Optional[List[str]] = None  # None=全保留；["A"]=仅 AIS Class A
    raw_data_root: str = ""  # 非空时覆盖 PathConfig.raw_data_dir（读取 ZIP）
    subset_output_root: str = ""  # 非空时按日 parquet 写入此根下 raw_daily_filtered（见 loader）
    # roi_bbox: (lon_min, lat_min, lon_max, lat_max) 粗筛（度）
    roi_bbox: Tuple[float, float, float, float] = (-118.5, 33.45, -118.0, 33.85)
    roi_polygon_path: str = ""  # 精筛多边形 GeoJSON（可为 outer ROI）
    # 可选：outer 与 corridor 之间的内港/核心水域多边形（空则跳过，清单 12）
    inner_roi_polygon_path: str = ""
    route_corridor_path: str = ""  # 主航路 corridor GeoJSON
    exclude_terminal_polygon_paths: List[str] = field(default_factory=list)
    exclude_anchorage_polygon_paths: List[str] = field(default_factory=list)
    timezone_local: str = "America/Los_Angeles"
    roi_version: str = "placeholder_v0"
    filter_profile: str = "default"


@dataclass
class PortCleaningConfig:
    """
    港口化清洗真源（清单 4、13–14）。
    运行流水线前应通过 sync_port_cleaning_to_pipeline 同步到 FilterConfig / TrajSplitConfig。
    """

    sog_max_knots: float = 25.0
    sog_max_knots_candidates: Tuple[float, ...] = (30.0, 35.0, 40.0)
    cog_invalid_values: Tuple[float, ...] = (360.0,)
    heading_invalid_values: Tuple[float, ...] = (511.0,)
    duplicate_time_tolerance_sec: float = 1.0
    gap_hours_for_split: float = 2.0
    berth_min_dwell_minutes: float = 30.0
    anchor_min_dwell_minutes: float = 15.0
    channel_low_speed_keep: float = 4.0  # 节：航道内低于此视为「航道低速保留」标注阈值


@dataclass
class ResampleExperimentConfig:
    """重采样消融（清单 25）。"""

    baseline_interval_sec: float = 120.0
    ablation_intervals_sec: Tuple[float, ...] = (60.0, 120.0)
    active_interval_sec: Optional[float] = None  # 非空则覆盖当前 run 的 resample_interval
    run_all_ablation_intervals: bool = False  # True 时离线脚本对多间隔各跑一遍 step5


@dataclass
class ClusterExperimentConfig:
    """聚类实验占位（清单 29–30，阶段三接线）。"""

    cluster_by_ship_group: bool = True
    cluster_by_direction: Optional[bool] = None
    min_cluster_size: int = 3
    manual_review_required: bool = True


@dataclass
class SplitConfig:
    """
    训练集划分（清单 36）。
    - time_based_split=False（默认）：按轨迹随机划分，与历史行为一致。
    - time_based_split=True 且 split_by_voyage=True：按每条轨迹 Timestamp.max() 的 UTC 日历年
      划入 train_years / val_years / test_years（互斥优先级：train > val > test）；
      未命中任何配置年份的轨迹丢弃并记日志。半年切分可后续扩展字段，当前仅日历年。
    """

    train_years: Tuple[int, ...] = (2023,)
    val_years: Tuple[int, ...] = ()
    test_years: Tuple[int, ...] = (2024,)
    split_by_voyage: bool = True
    no_leakage_by_mmsi_window: bool = True
    time_based_split: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤1：数据加载配置
# ═══════════════════════════════════════════════════════════════════════════════


def _default_marine_raw_columns() -> List[str]:
    """清单 8：保留列（Marine Cadastre 常用）。"""
    return [
        "MMSI",
        "BaseDateTime",
        "LAT",
        "LON",
        "SOG",
        "COG",
        "Heading",
        "VesselType",
        "Status",
        "TransceiverClass",
        "Length",
        "Width",
        "Draft",
    ]


@dataclass
class DataLoadConfig:
    """步骤1 - 原始AIS数据加载相关配置"""

    raw_columns: List[str] = field(default_factory=_default_marine_raw_columns)
    column_rename: dict = field(
        default_factory=lambda: {
            "BaseDateTime": "Timestamp",
        }
    )
    timestamp_format: str = "%Y-%m-%dT%H:%M:%S"
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    csv_chunksize: int = 200_000
    keep_base_datetime_copy: bool = True  # 重命名后保留 BaseDateTime 副本用于审计


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤2：异常值过滤配置
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FilterConfig:
    """步骤2 - 逐条AIS报文异常值检测与过滤参数"""

    lon_min: float = 113.5
    lon_max: float = 115.5
    lat_min: float = 29.8
    lat_max: float = 31.2

    sog_min: float = 0.0
    sog_max: float = 25.0

    cog_min: float = 0.0
    cog_max: float = 360.0
    heading_unavailable: float = 511.0
    cog_unavailable: float = 360.0

    jump_speed_factor: float = 1.5
    knot_to_ms: float = 0.5144
    earth_radius: float = 6371000.0

    duplicate_time_threshold: float = 1.0
    # Marine Cadastre 域（LB_LA_ADAPTATION_PLAN §4.1 domain_adapted）：
    # True：COG=360 置 NaN 后不因 COG 缺失删行；False：对齐方案 2.5 字面「剔除无可用 COG 行」。
    # 每次实验须在 EXPERIMENT_LOG_TEMPLATE「步骤2：COG」中记录本开关，避免复现口径漂移。
    allow_missing_cog_rows: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤3：停泊检测配置
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BerthConfig:
    """步骤3 - 停泊点检测与停泊数据段删除参数（方案窗口扩展语义保留）。"""

    distance_threshold: float = 50.0
    speed_threshold: float = 0.5
    time_threshold: float = 1800.0
    mode: str = "delete"  # "delete" | "label_only"
    terminal_polygon_paths: List[str] = field(default_factory=list)
    anchorage_polygon_paths: List[str] = field(default_factory=list)
    channel_corridor_polygon_path: Optional[str] = None
    label_terminal_sog_max: float = 3.0
    label_anchorage_sog_max: float = 3.0
    # 由 sync_port_cleaning_to_pipeline 从 PortCleaningConfig.channel_low_speed_keep 写入（单一真源）
    channel_low_speed_for_label: float = 4.0


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤4：轨迹切分配置
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrajSplitConfig:
    """步骤4 - 航行轨迹切分参数"""

    time_gap: float = 7200.0
    min_traj_length: int = 20
    log_short_traj_drop_ratio: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤5：插值重采样配置
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ResampleConfig:
    """步骤5 - 线性插值与等时间间隔重采样参数"""

    resample_interval: float = 120.0
    # omit | forward_fill | circular（清单 26–27）
    resample_heading_mode: str = "omit"
    output_subdir: str = ""  # 相对 resampled_dir，如 dt120


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤6：CDDTW聚类配置
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ClusterConfig:
    """步骤6 - CDDTW聚类全部参数"""

    dbscan_eps: float = 0.001
    dbscan_min_samples: int = 5
    ball_tree_leaf_size: int = 3

    split_sigma_factor: float = 2.0
    max_refine_depth: int = 5
    min_cluster_size: int = 3

    merge_threshold: Optional[float] = None
    merge_auto_sigma_factor: float = 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤7：归一化配置
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class NormalizationConfig:
    """步骤7 - Min-Max归一化参数"""

    epsilon: float = 1e-8
    features: List[str] = field(
        default_factory=lambda: ["LON", "LAT", "SOG", "COG"]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤8 & 9：LSTM训练配置
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainConfig:
    """步骤8~9 - LSTM模型训练参数"""

    default_time_steps: int = 12
    curvature_to_T: dict = field(
        default_factory=lambda: {
            (0.0, 1.05): 14,
            (1.05, 1.45): 12,
            (1.45, 2.0): 10,
            (2.0, float("inf")): 8,
        }
    )
    curvature_to_H: dict = field(
        default_factory=lambda: {
            (0.0, 1.05): 88,
            (1.05, 1.35): 108,
            (1.35, 1.45): 128,
            (1.45, 2.0): 168,
            (2.0, float("inf")): 188,
        }
    )
    default_hidden_size: int = 128

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    batch_size: int = 128
    epochs: int = 500
    learning_rate: float = 0.001
    dropout: float = 0.2

    early_stop_patience: int = 40
    lr_scheduler_patience: int = 15
    lr_scheduler_factor: float = 0.5

    input_features: int = 4
    output_features: int = 4


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤10~12：在线预测配置
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PredictConfig:
    """步骤10~12 - 在线预测参数"""

    prediction_steps: int = 15
    buffer_timeout: float = 7200.0
    # 单船滑动缓冲最大点数（步骤10）；加载离线资产后会与各簇模型 T 取较大值并打日志
    buffer_capacity_steps: int = 14
    # 11.3：None 时用 outputs/clusters/match_unknown_threshold.json 标定值，否则 unknown_threshold_fallback（须在实验记录中写明来源）
    unknown_threshold: Optional[float] = None
    unknown_threshold_fallback: float = 0.15
    max_fork_branches: int = 3
    fork_deviation_threshold: float = 10.0
    # 方案 11.4 第四步：跨多次预测调用，用新观测与上一时刻各分支首步预测比距，累积后锁定航路
    fork_disambiguation_min_observations: int = 2
    fork_disambiguation_error_ratio: float = 1.35
    fork_disambiguation_enabled: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# 同步与汇总
# ═══════════════════════════════════════════════════════════════════════════════


def domain_focus_spatial_active(domain: "DomainFocusConfig") -> bool:
    """清单 5 / 计划：enabled 或 subset_output_root 非空时走 LA/LB 空间与廉价域过滤。"""
    return bool(domain.enabled) or bool(str(domain.subset_output_root or "").strip())


def sync_port_cleaning_to_pipeline(config: "RSTPMConfig") -> None:
    """将 PortCleaningConfig 同步到 FilterConfig / TrajSplitConfig（单一真源，清单 4、22）。"""
    pc = config.port_cleaning
    config.filter.sog_max = float(pc.sog_max_knots)
    config.filter.duplicate_time_threshold = float(pc.duplicate_time_tolerance_sec)
    if pc.cog_invalid_values:
        config.filter.cog_unavailable = float(pc.cog_invalid_values[0])
    if pc.heading_invalid_values:
        config.filter.heading_unavailable = float(pc.heading_invalid_values[0])
    config.traj_split.time_gap = float(pc.gap_hours_for_split) * 3600.0
    # 清单 4 channel_low_speed_keep 与步骤 3 航道低速标注阈值单一来源
    config.berth.channel_low_speed_for_label = float(pc.channel_low_speed_keep)


def apply_domain_roi_to_filter(config: "RSTPMConfig") -> None:
    """domain_focus 激活（enabled 或 subset_output_root）时将粗 bbox 写入 FilterConfig。"""
    if not domain_focus_spatial_active(config.domain_focus):
        return
    lon_min, lat_min, lon_max, lat_max = config.domain_focus.roi_bbox
    config.filter.lon_min = lon_min
    config.filter.lon_max = lon_max
    config.filter.lat_min = lat_min
    config.filter.lat_max = lat_max


def apply_resample_experiment_interval(config: "RSTPMConfig") -> None:
    """根据 ResampleExperimentConfig 设置当前 resample_interval 与输出子目录（清单 25）。"""
    re = config.resample_experiment
    if re.active_interval_sec is not None:
        interval = float(re.active_interval_sec)
    else:
        interval = float(re.baseline_interval_sec)
    config.resample.resample_interval = interval
    config.resample.output_subdir = f"dt{int(interval)}"


@dataclass
class RSTPMConfig:
    """RSTPM 项目全局配置汇总"""

    paths: PathConfig = field(default_factory=PathConfig)
    domain_focus: DomainFocusConfig = field(default_factory=DomainFocusConfig)
    port_cleaning: PortCleaningConfig = field(default_factory=PortCleaningConfig)
    resample_experiment: ResampleExperimentConfig = field(
        default_factory=ResampleExperimentConfig
    )
    cluster_experiment: ClusterExperimentConfig = field(
        default_factory=ClusterExperimentConfig
    )
    split: SplitConfig = field(default_factory=SplitConfig)
    data_load: DataLoadConfig = field(default_factory=DataLoadConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    berth: BerthConfig = field(default_factory=BerthConfig)
    traj_split: TrajSplitConfig = field(default_factory=TrajSplitConfig)
    resample: ResampleConfig = field(default_factory=ResampleConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    random_seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    verbose: bool = True


def get_default_config() -> RSTPMConfig:
    cfg = RSTPMConfig()
    sync_port_cleaning_to_pipeline(cfg)
    return cfg


RSTMPConfig = RSTPMConfig
