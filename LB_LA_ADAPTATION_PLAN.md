# LA/LB 领域适配说明（清单 48）

## 1. 为何不用论文原研究区

`RSTPM_full_pipeline.md` 步骤 2.3 中的武汉段 / 浙江内河边界为**论文示例**。本仓库第一阶段目标为 **Los Angeles / Long Beach 港域及主航路** 的商船短时多步轨迹预测，研究区围绕 San Pedro Bay、Cerritos Channel、breakwater 内港及进出港主航路，需在配置与 GeoJSON 中单独定义（见 `DomainFocusConfig`、`resources/*.geojson`）。

## 2. 任务边界（清单 1）

- **第一阶段**：离线历史 AIS 轨迹建模（步骤 1–9）；在线步骤 10–12 在离线稳定后再接。
- **研究问题**：LA/LB 主要商船航路的短时多步轨迹预测（状态变量以 `LON/LAT/SOG/COG` 为主）。
- **保留船型**：`VesselType ∈ [70,79]`（Cargo）、`[80,89]`（Tanker），由 `vessel_type_keep` / `ship_groups` 配置。
- **AIS Class A/B**：不在代码中写死；通过 `transceiver_class_keep = None` 或 `["A"]` 等做**对照实验**，结论写入实验记录。

## 3. 时区（清单 3）

- 内部全流程以 **UTC** 为准（Marine Cadastre `BaseDateTime` 为 UTC）。
- 仅在 EDA / 可视化 / 业务解释时使用 `America/Los_Angeles`（`DomainFocusConfig.timezone_local`）。
- 跨自然日轨迹拼接以 UTC 连续时间为准，不按本地日历强切。

## 4. 与论文/真源文本的差异（domain_adapted，避免静默漂移）

### 4.1 COG = 360、Heading = 511（清单 13）

- **论文方案步骤 2.5**：`COG` 缺失或越界可删行。
- **Marine Cadastre**：`COG = 360.0`、`Heading = 511` 表示**不可用**，非真实角度。
- **本仓库港口化实现**：在步骤 2 中将上述哨兵**置为 NaN**，**不因 COG 缺失而删除整行**（与 `RSTPM_full_pipeline.md` 2.5 字面不同，属 **domain_adapted**；重采样阶段对 NaN 的 `COG` 需用邻段插值或前向填充，见 `TrajectoryResampler`）。

### 4.2 停泊段（清单 17–20）

- **方案步骤 3**：长时停泊窗口删除仍为默认能力（`BerthConfig.mode == "delete"`）。
- **港口场景**：增加 `label_only` 模式，先打标 `is_terminal_dwell` / `is_anchorage_wait` / `is_channel_low_speed` 再人工验收，避免误删港池低速与排队航段。

## 5. 主输入维度（清单 27–28）

- LSTM 主输入仍为 **4 维**：`LON, LAT, SOG, COG`。
- `Heading`、`Length`、`Width`、`Draft`、`VesselType` 等保留在宽表中用于过滤、分层与后续增强特征，**第一版不强制进入 LSTM 输入**。

## 6. 参数与实验

- 论文附录 A 中的数值多为 **起点**；港口场景下 `V_max`、`L_min`、`ΔT` 等见 `PARAM_LAYERS.md` 与 `PortCleaningConfig` / `ResampleExperimentConfig`。
- 每次离线跑数应填写 `EXPERIMENT_LOG_TEMPLATE.md` 并保存 `run_config_snapshot.json`（若流水线写入）。

## 7. TransceiverClass 策略（清单 16，文档写死）

- **统计入口**：`subset_manifest_*.json` 中的 `transceiver_class_dist`；建议在 EDA 中补充 Cargo/Tanker 分层下的 A/B 占比与平均采样间隔。
- **实现策略**（三选一，由 `DomainFocusConfig.transceiver_class_keep` 与实验记录共同固定，避免静默漂移）：
  1. `None`：A+B 全保留；
  2. `["A"]`：仅 Class A；
  3. 未来可扩展：A/B **分开建模**（离线聚类与训练各跑一套资产）。
- **禁止**：在 `step2` 中仅凭 `Status` 或 `TransceiverClass` 删除整条轨迹（清单 15）。
