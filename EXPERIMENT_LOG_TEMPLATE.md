# 实验记录模板（清单 50）

## 元信息

- **实验 ID**：
- **日期**：
- **操作者**：
- **git commit**（或版本标签）：

## 数据与领域

- **数据范围**（日期 / ZIP 列表）：
- **船型范围**（如 70–89、仅 A 类等）：
- **ROI 版本**（`geo_asset_manifest.json` 中 version）：
- **filter_profile**（如 `la_lb_merchant_v1`）：

## 流水线参数

- **重采样间隔（秒）**：
- **训练/验证/测试划分策略**（若已启用 `SplitConfig`）：
- **聚类参数摘要**（R、MinP、min_cluster_size 等）：
- **模型参数**（T、H、batch、lr、dropout、epochs）：

## 步骤 2：COG / 哨兵策略（清单 13 / LB_LA domain_adapted）

- **`FilterConfig.allow_missing_cog_rows`**（本次运行 True/False）：
- **含义**：`True` 时 Marine Cadastre `COG=360` 置 NaN 后**不因 COG 缺失删除整行**；`False` 时对齐方案字面「缺失则从主训练序列剔除」。
- **与重采样衔接**：缺失 COG 在 `TrajectoryResampler` 中的填充策略（见运行日志 / 配置快照）。

## 在线阈值与标定（步骤 11.3）

- **`predict.unknown_threshold`**：若显式设置，记录数值；若为 `null`，说明使用 `outputs/clusters/match_unknown_threshold.json` 中 `unknown_threshold_suggested` 或 fallback。
- **`unknown_threshold_fallback` 实际取值**（来自 `run_config_snapshot.json` 或命令行）：
- **`match_unknown_threshold.json` 生成 commit / 日期**（若使用标定文件）：

## 结果摘要

- **主要指标**（MSE/MAE/位置误差/多步误差等）：
- **删除比例**（各过滤步）：
- **轨迹条数 / 样本数**：

## 结论与下一步

- **结论**：
- **下一步动作**：
