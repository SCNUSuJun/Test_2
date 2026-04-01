# RSTPM - 基于AIS数据的实时船舶轨迹预测模型

## 项目概述

RSTPM (Real-time Ship Trajectory Prediction Model) 是一个完整的船舶轨迹预测系统，
基于 AIS (Automatic Identification System) 数据，通过 CDDTW 聚类 + 多 LSTM 模型
实现对船舶未来 30 分钟轨迹的实时预测。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        离线训练阶段 (步骤1→9)                         │
│                                                                     │
│  AIS_*.zip ──→ 步骤1:加载 ──→ 步骤2:过滤 ──→ 步骤3:去停泊            │
│               排序          5条规则       滑动窗口检测               │
│                                                                     │
│  ──→ 步骤4:切分 ──→ 步骤5:插值 ──→ 步骤6:CDDTW聚类                  │
│     时间间隔>2h    等间隔120s    6A起终点 → 6B DBSCAN                │
│                                  → 6C DTW细化 → 6D合并              │
│                                                                     │
│  ──→ 步骤7:归一化 ──→ 步骤8:构造样本 ──→ 步骤9:LSTM训练              │
│     每类别Min-Max    滑动窗口T×4→1×4   每类别独立模型               │
│                                                                     │
│  输出: N个LSTM模型 + N组归一化参数 + N条特征轨迹                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        在线预测阶段 (步骤10→12)                       │
│                                                                     │
│  实时AIS流 ──→ 步骤10:预处理 ──→ 步骤11:类别匹配 ──→ 步骤12:推理     │
│              缓冲区管理    DTW滑窗匹配+Top-K近邻启发式   多步递推预测   │
│              实时插值    可选多分支预测与跨观测消歧(11.4④) 反归一化   │
│                                                                     │
│  输出: 未来30分钟(15步)预测轨迹 (lon, lat, sog, cog, t)             │
└─────────────────────────────────────────────────────────────────────┘
```

**在线能力边界（默认 `run_online_prediction.py`）**：

- **类别匹配**：步骤 11.3 的改进 DTW 滑窗匹配 + `match_unknown_threshold`（可离线标定）已实现。
- **多航路分流（方案 11.4）**：在「前两名匹配距离接近」时进入多模型并行预测；**第四步**（随新 AIS 累积比较各分支上一时刻首步预测与当前位置、锁定航路）由 `online/fork_disambiguation.py` + `OnlinePredictionSystem` 内状态机实现，配置见 `PredictConfig.fork_disambiguation_*`。
- **航道几何剪枝**：若存在可读且含 LineString 的 `resources/channel_centerlines.geojson`（或通过 `--channel_geojson` 指定），则加载 `GeoJsonLineStringChannelStore` 并参与 `prune_branch_predictions_by_geometry`；否则与过去行为一致，几何剪枝不生效。
- **分叉口空间注册**：`resources/fork_junctions.json` 中 `junctions` 非空时加载 `JsonForkJunctionRegistry`；**当前默认文件为空列表**，等价于未注册分叉口（与 `NullForkJunctionRegistry` 行为一致）。与方案文中「接近分叉口再触发」结合的严格空间门控，需在 JSON 中填入圆域与 `cluster_ids` 后，再在 `detect_fork_situation` 中与 `junction_for_location` 联动扩展（见 `resources/implementations.py` 注释）。

## 目录结构

```
RSTPM/
├── config/                          # 配置模块
│   ├── __init__.py
│   └── settings.py                  # 全局配置（所有超参数集中管理）
│
├── data_processing/                 # 数据处理模块（步骤1~5）
│   ├── __init__.py
│   ├── step1_data_loader.py         # 步骤1: AIS数据加载与排序
│   ├── step1b_domain_filter.py      # 步骤1b: LA/LB 多边形/corridor 空间精筛
│   ├── step2_anomaly_filter.py      # 步骤2: 5条规则异常值过滤
│   ├── step3_berth_detection.py     # 步骤3: 停泊段检测与删除
│   ├── step4_trajectory_split.py    # 步骤4: 轨迹切分与短轨迹过滤
│   └── step5_resample.py           # 步骤5: 线性插值等间隔重采样
│
├── clustering/                      # 聚类模块（步骤6）
│   ├── __init__.py
│   ├── step6a_endpoints.py          # 步骤6A: 提取轨迹起终点
│   ├── step6b_dbscan.py            # 步骤6B: Ball-Tree DBSCAN聚类
│   ├── step6c_dtw_refine.py        # 步骤6C: 改进DTW轨迹细化
│   └── step6d_merge.py             # 步骤6D: 跨类别合并 + 最终输出
│
├── model/                           # 模型模块（步骤7~9）
│   ├── __init__.py
│   ├── step7_normalization.py       # 步骤7: Min-Max归一化
│   ├── step8_sample_construction.py # 步骤8: 滑动窗口样本构造
│   ├── step9_lstm_train.py          # 步骤9: LSTM模型定义与训练
│   └── evaluation.py               # 评估指标计算
│
├── online/                          # 在线预测模块（步骤10~12）
│   ├── __init__.py
│   ├── step10_realtime_preprocess.py # 步骤10: 实时数据预处理与缓冲区
│   ├── step11_cluster_match.py      # 步骤11: 聚类类别匹配
│   ├── step12_prediction.py         # 步骤12: LSTM推理与多步预测
│   └── fork_disambiguation.py       # 方案11.4④ 跨调用分叉消歧状态
│
├── utils/                           # 工具函数
│   ├── __init__.py
│   └── common.py                    # Haversine/角度插值/弯曲系数/IO
│
├── schemas/                         # 跨模块数据契约（列名、资产、在线输出结构）
│   ├── __init__.py
│   ├── constants.py
│   └── assets.py
│
├── resources/                       # 外部资源抽象 + LA/LB GeoJSON（多分叉几何、MMSI 统计等）
│   ├── __init__.py
│   ├── fork_resources.py
│   ├── implementations.py           # Null / GeoJSON 中心线 / JSON 分叉口注册加载器
│   ├── geo_asset_manifest.json
│   ├── fork_junctions.json          # 方案11.4 分叉口圆域与 cluster_ids（默认可为空）
│   └── *.geojson                    # ROI / corridor / channel_centerlines 等
│
├── scripts/                         # 运行脚本
│   ├── run_offline_training.py      # 离线训练全链路（步骤1→9）
│   └── run_online_prediction.py     # 在线预测系统（步骤10→12）
│
├── tests/                           # 测试用例
│   ├── __init__.py
│   ├── test_core.py                 # 工具 / DTW / LSTM / 归一化等单元测试
│   ├── test_minimal_pipeline.py    # 合成数据 2→5 与聚类 6A→6B 烟测
│   ├── test_evaluation.py          # 附录 B 评估指标单测
│   ├── test_la_lb_phase12.py       # LA/LB 阶段 1–2 最小集
│   └── test_round2_plan.py         # 第二轮：在线网格缓冲、按年划分等
│
├── RSTPM_full_pipeline.md          # 方案入口（指向 surgical_fix 全文）
├── RSTPM_full_pipeline_surgical_fix_LALB.md  # 12步全文（LA/LB 基准）
├── 方案3.md                         # 真源索引（链接全文与 LB_LA 适配）
├── outputs/                         # 输出目录
│   ├── intermediate/                # 各步骤中间产物
│   ├── models/                      # 训练好的LSTM模型
│   ├── clusters/                    # 聚类结果（含 final_clusters.pkl、match_unknown_threshold.json、mmsi_cluster_counts.json 等）
│   ├── normalization/               # 归一化参数
│   ├── logs/                        # 训练日志
│   ├── figures/                     # 可视化图表
│   └── subsets/                     # LA/LB 子集：raw_daily_filtered、metadata、merged_monthly
│
├── requirements.txt                 # 开发与 CI 完整依赖（以此为准）
├── pyproject.toml                   # 可编辑安装的最小运行时依赖声明
└── README.md
```

## 数据集

AIS 日频数据，731个ZIP文件（2023-01-01 ~ 2024-12-31），约 227 GB。

每日 CSV 常含 17 列（Marine Cadastre 等）:
`MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, IMO, CallSign, VesselType, Status, Length, Width, Draft, Cargo, TransceiverClass`

默认管线在 `DataLoadConfig.raw_columns` 中保留 **13 列**（含 Heading、VesselType、Status、TransceiverClass、尺寸等），主流程仍以 **MMSI / Timestamp / LON / LAT / SOG / COG** 为核心；列集合可在 `config/settings.py` 调整。

## 运行环境（推荐）

在项目根目录执行可编辑安装，使 `config`、`data_processing` 等包可被任意工作目录下的脚本导入：

```bash
pip install -r requirements.txt
pip install -e .
```

**依赖声明**：日常开发、CI 与可复现环境以 **`requirements.txt`** 为完整依赖清单；**`pyproject.toml`** 仅服务 `pip install -e .` 的最小运行时集合，二者条目可能不完全一致。

若不安装包，也可在每次运行前设置（等价于脚本内 `sys.path` 插入根目录）：

```bash
export PYTHONPATH="/absolute/path/to/RSTPM:$PYTHONPATH"
```

配置项含义与「论文默认值 / 数据集已验证 / 待调参」分层说明见 [config/settings.py](config/settings.py) 顶部文档与字段旁注释。

**大规模数据（约 227GB）**：步骤 1 不应默认一次性拼接全期数据进内存；应按日/按 zip 落盘至 `outputs/intermediate/step1_by_day/`，并用 `pipeline_checkpoint.json` 做断点（约定见 [data_processing/step1_data_loader.py](data_processing/step1_data_loader.py)）。

## 快速开始

### 1. 安装依赖

见上文「运行环境（推荐）」：`pip install -r requirements.txt` 与 `pip install -e .`。

### 2. 离线训练
```bash
python scripts/run_offline_training.py \
    --raw_data_dir /path/to/AIS_zips \
    --date_start 2023-01-01 \
    --date_end 2023-06-30 \
    --device cuda
```

可选 `--config overrides.json`：嵌套键与 `RSTPMConfig` 子 dataclass 字段同名；命令行参数在解析完成后再次覆盖 JSON。

### 3. 在线预测（模拟模式）
```bash
python scripts/run_online_prediction.py \
    --mode simulate \
    --data_file /path/to/AIS_2024_01_01.csv
```

可选：`--channel_geojson` / `--fork_junctions_json` 覆盖默认资源路径；未指定且仓库内存在 `resources/channel_centerlines.geojson`（含 LineString）时会自动加载几何剪枝。

## 关键参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| 地理范围 | 因区域而异 | 步骤2规则1 |
| V_max | 25节 | 最大合理航速 |
| 跳跃安全系数k | 1.5 | 步骤2规则4 |
| 停泊距离阈值 | 50米 | 步骤3 |
| 停泊速度阈值 | 0.5节 | 步骤3 |
| 轨迹切分间隔 | 2小时 | 步骤4 |
| 最小轨迹长度 | 20点 | 步骤4 |
| 重采样间隔 | 120秒 | 步骤5 |
| DBSCAN R | 0.001 | 步骤6B |
| DBSCAN MinP | 5 | 步骤6B |
| 时间步长T | 8~14 | 依弯曲系数 |
| 隐藏层H | 88~188 | 依弯曲系数 |
| 批量大小 | 128 | 步骤9 |
| 学习率 | 0.001 | Adam |
| Dropout | 0.2 | 步骤9 |
| 早停patience | 40 | 步骤9 |

## 精度基准

- 非分叉河段: 单条ADE≈18.4m, MDE<60m
- 多分叉河段: ADE 14~52m
- 12分钟内: ADE < 100m
- 30分钟内: ADE < 500m
