# 参数分层（清单 49）

与 [`config/settings.py`](config/settings.py) 字段交叉引用。层级标签：**paper_fixed** | **domain_adapted_default** | **to_be_tuned** | **deprecated / not_used**。

## paper_fixed（与 RSTPM 论文/方案结构绑定，不改语义则不改名）

| 含义 | 配置入口 |
|------|----------|
| 五类异常过滤结构（地理/航速/航向/跳跃/重复） | `FilterConfig` + `AnomalyFilter` |
| 停泊窗口：空间阈值 + 低速阈值 + 时间跨度 | `BerthConfig.distance_threshold` 等 |
| 轨迹按时间间隔切分、最小长度 | `TrajSplitConfig` |
| LON/LAT/SOG 线性插值、COG 圆周插值 | `TrajectoryResampler` + `interpolate_cog` |
| CDDTW：Ball-Tree DBSCAN + 改进 DTW + 合并 | `ClusterConfig` |
| 按簇归一化、每簇 LSTM | `NormalizationConfig`、`TrainConfig` |

## domain_adapted_default（LA/LB / Marine Cadastre 默认）

| 含义 | 配置入口 |
|------|----------|
| 研究区 bbox、polygon、corridor、排除区路径 | `DomainFocusConfig` |
| COG=360 / Heading=511 置空、港口航速上限、重复时间容差、切分小时 | `PortCleaningConfig`（同步到 `FilterConfig` / `TrajSplitConfig`，见 `sync_port_cleaning_to_pipeline`） |
| 子集输出根目录、按日 parquet / manifest | `DomainFocusConfig.subset_output_root` + `AISDataLoader` |
| 停泊 `delete` / `label_only`、终端/锚地/航道几何 | `BerthConfig` |
| 重采样 baseline 与消融间隔 | `ResampleExperimentConfig` + `ResampleConfig` |
| 聚类按船型/方向（后续阶段接线） | `ClusterExperimentConfig` |
| 按年划分训练/验证/测试（后续阶段接线） | `SplitConfig` |

## to_be_tuned

| 含义 | 配置入口 |
|------|----------|
| DBSCAN R/MinP、DTW 分裂/合并阈值 | `ClusterConfig` |
| 未知类别匹配阈值 | `PredictConfig.unknown_threshold` |
| `sog_max` 港口候选 30/35/40 节对比 | `PortCleaningConfig.sog_max_knots_candidates`（仅日志统计） |
| 实际采用的 `L_min`、重采样间隔 60 vs 120 s | 实验后写入 `ResampleConfig` / `TrajSplitConfig` |

## deprecated / not_used

| 说明 |
|------|
| 若仅使用 `PortCleaningConfig` 同步后的值，避免在业务脚本中再写一份 `FilterConfig.sog_max` 手工覆盖（除非有意做消融）。 |
| `config.RSTMPConfig` 为历史别名，新代码请用 `RSTPMConfig`。 |
