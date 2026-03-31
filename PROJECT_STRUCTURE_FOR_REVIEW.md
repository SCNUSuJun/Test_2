# RSTPM 项目结构说明（供 ChatGPT / 人工审查速览）

> 业务真源：**[RSTPM_full_pipeline.md](RSTPM_full_pipeline.md)**（全链路 12 步）。  
> 工程约束：**[.cursorrules](.cursorrules)**。  
> **[方案3.md](方案3.md)**：索引 `RSTPM_full_pipeline.md` 与 **[LB_LA_ADAPTATION_PLAN.md](LB_LA_ADAPTATION_PLAN.md)**。  
> 参数分层：**[PARAM_LAYERS.md](PARAM_LAYERS.md)**；实验模板：**[EXPERIMENT_LOG_TEMPLATE.md](EXPERIMENT_LOG_TEMPLATE.md)**。

**依赖**：开发与 CI 以根目录 **`requirements.txt`** 为完整依赖清单；**`pyproject.toml`** 为可编辑安装的最小运行时声明，集合未必逐项相同（见 README）。

---

## 一、建议阅读顺序（按数据流）

1. **真源与规则**  
   - `RSTPM_full_pipeline.md` → 算法与参数定义  
   - `.cursorrules` → 不可静默漂移的硬性约定  

2. **配置单一入口**  
   - `config/settings.py` → 全部超参数、路径、`PredictConfig` / `ClusterConfig` 等  

3. **离线流水线（步骤 1→9）**  
   - 入口：`scripts/run_offline_training.py`（串起 1→9）  
   - 步骤 1–5：`data_processing/step1_*.py` … `step5_*.py`  
   - 步骤 6：`clustering/step6a_*.py` … `step6d_merge.py`  
   - 步骤 7–9：`model/step7_normalization.py`、`step8_sample_construction.py`、`step9_lstm_train.py`  

4. **在线流水线（步骤 10→12）**  
   - 入口：`scripts/run_online_prediction.py`  
   - `online/step10_realtime_preprocess.py` → `step11_cluster_match.py` → `step12_prediction.py`  

5. **契约与资源抽象**  
   - `schemas/`（列名、资产、在线输出结构）  
   - `resources/`（多分叉：几何 / MMSI 统计 / 分叉口接口与默认实现）  

6. **工具与离线标定**  
   - `utils/common.py`（Haversine、COG 插值、弯曲系数等）  
   - `utils/match_calibration.py`（未知类别阈值离线估计，供步骤 11 加载）  

7. **测试**  
   - `tests/test_core.py`、`test_minimal_pipeline.py`、`test_evaluation.py`  

---

## 二、目录树（逻辑分组）

```
RSTPM/
├── RSTPM_full_pipeline.md      # 方案全文（审查基准）
├── 方案3.md                    # 若有：真源索引，指向 RSTPM_full_pipeline.md
├── .cursorrules                # AI/开发硬性约定
├── README.md                   # 人类上手说明
├── PROJECT_STRUCTURE_FOR_REVIEW.md  # 本文件
├── pyproject.toml              # 包元数据与最小运行时依赖
├── requirements.txt            # 开发/CI 完整依赖（优先以此为准）
│
├── config/                     # 全局配置
│   ├── __init__.py
│   └── settings.py             # PathConfig / FilterConfig / ClusterConfig / TrainConfig / PredictConfig …
│
├── data_processing/            # 步骤 1–5
│   ├── step1_data_loader.py
│   ├── step1b_domain_filter.py  # LA/LB 业务域过滤（船型/ROI/corridor，清单 5）
│   ├── step2_anomaly_filter.py
│   ├── step3_berth_detection.py
│   ├── step4_trajectory_split.py
│   └── step5_resample.py
│
├── clustering/                 # 步骤 6（CDDTW）
│   ├── step6a_endpoints.py     # 起终点
│   ├── step6b_dbscan.py        # Ball-Tree + DBSCAN（sklearn）
│   ├── step6c_dtw_refine.py    # 改进 DTW 细化
│   └── step6d_merge.py         # 跨类合并、特征轨迹与 orientation_flip
│
├── model/                      # 步骤 7–9 + 评估
│   ├── step7_normalization.py
│   ├── step8_sample_construction.py
│   ├── step9_lstm_train.py
│   └── evaluation.py           # 附录 B 类指标
│
├── online/                     # 步骤 10–12
│   ├── step10_realtime_preprocess.py
│   ├── step11_cluster_match.py
│   └── step12_prediction.py
│
├── utils/
│   ├── common.py
│   └── match_calibration.py    # match_unknown_threshold.json 标定逻辑
│
├── schemas/                    # 数据契约
│   ├── constants.py
│   └── assets.py               # ClusterAsset、ModelBundle、OnlinePredictionResult 等
│
├── resources/                  # 多分叉等外部资源抽象 + LA/LB GeoJSON（清单 10–12）
│   ├── fork_resources.py       # ABC 接口
│   ├── implementations.py      # 内存统计、Null 几何、几何剪枝钩子
│   ├── geo_asset_manifest.json # 空间资源清单（清单 46）
│   └── *.geojson               # ROI / corridor / exclusion 占位几何
│
├── scripts/
│   ├── run_offline_training.py # 离线全链路
│   └── run_online_prediction.py
│
├── tests/
│   ├── test_core.py
│   ├── test_minimal_pipeline.py
│   ├── test_evaluation.py
│   ├── test_la_lb_phase12.py
│   └── test_round2_plan.py
│
└── outputs/                    # 运行产物目录（通常不入库或为空）
    ├── subsets/                # LA/LB 子集：raw_daily_filtered、metadata、merged_monthly（清单 9）
    ├── intermediate/
    ├── clusters/               # final_clusters.pkl、cluster_manifest.json、
    │                           # match_unknown_threshold.json、mmsi_cluster_counts.json …
    ├── models/                 # lstm_cluster_*.pt、model_bundle_cluster_*.json
    ├── normalization/          # norm_params_cluster_*.json
    └── logs/
```

---

## 三、关键产物与在线加载关系（审查闭环时常查）

| 产物（相对 `outputs/`） | 写入方（典型） | 读取方（典型） |
|-------------------------|----------------|----------------|
| `clusters/final_clusters.pkl` | `run_offline_training.py` | `ClusterMatcher`（无 bundle 时补特征轨迹） |
| `clusters/cluster_manifest.json` | 同上 | 人类/工具；含 `auxiliary_artifacts` 指辅助文件 |
| `clusters/match_unknown_threshold.json` | 同上（标定） | `ClusterMatcher.load_cluster_assets` |
| `clusters/mmsi_cluster_counts.json` | 同上 | `InMemoryMmsiBranchStatsStore.load_from_offline_json` |
| `clusters/feature_traj_cluster_*.npz` | 同上 | bundle + 在线匹配 |
| `normalization/norm_params_cluster_*.json` | step7 | 在线归一化/反归一化 |
| `models/lstm_cluster_*.pt` | step9 | 在线推理 |
| `models/model_bundle_cluster_*.json` | 同上 | 在线优先加载路径 |

---

## 四、压缩包说明

根目录下的 **`RSTPM-ChatGPT-review.zip`**（若已生成）一般已排除：

- `__pycache__/`、`*.pyc`
- `.pytest_cache/`
- `rstpm.egg-info/`
- `.git/`（若存在）

**未排除**源代码、`RSTPM_full_pipeline.md`、`.cursorrules`、`outputs/` 空目录结构等。若需带上大数据产物，请自行按需打包 `outputs/` 下实际文件。

---

## 五、给审查者的一句话

先对照 **RSTPM_full_pipeline.md** 逐步编号，用 **`scripts/run_offline_training.py` / `run_online_prediction.py`** 核对调用链，用 **`config/settings.py`** 核对参数单一来源，用 **`schemas/assets.py`** 核对在线输出字段（如 `points` 与 `t`）。
