"""
离线训练全链路脚本（步骤1 → 步骤9）
=====================================

将离线训练阶段的所有步骤串联为一条完整的流水线：
  步骤1: 加载原始AIS数据
  步骤2: 异常值过滤（5条规则）
  步骤3: 停泊段检测与删除
  步骤4: 轨迹切分
  步骤5: 线性插值重采样
  步骤6: CDDTW聚类（6A→6B→6C→6D；方案步骤6.5/6E 的最终输出由 step6d_merge 完成）
  步骤7: 数据归一化
  步骤8: 构造LSTM训练样本
  步骤9: LSTM训练

每一步的输出都是下一步的输入，严格无断层。

用法：
    python scripts/run_offline_training.py [选项]
    python scripts/run_offline_training.py --config path/to/overrides.json

--config 为可选 JSON：键与 RSTPMConfig 顶层字段一致，嵌套对象对应子 dataclass；
命令行参数在解析完成后再次覆盖对应字段。
"""

import sys
import os
import time
import json
import subprocess
import argparse
from dataclasses import asdict, is_dataclass

import numpy as np

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    get_default_config,
    RSTPMConfig,
    sync_port_cleaning_to_pipeline,
    apply_domain_roi_to_filter,
    apply_resample_experiment_interval,
    domain_focus_spatial_active,
)
from schemas.assets import ModelBundleDescriptor
from data_processing import (
    AISDataLoader,
    AnomalyFilter,
    BerthDetector,
    TrajectorySplitter,
    TrajectoryResampler,
)
from data_processing.step1b_domain_filter import DomainSpatialFilter
from clustering import (
    EndpointExtractor,
    StartEndClusterer,
    TrajectoryRefiner,
    ClusterMerger,
)
from model import (
    DataNormalizer,
    SampleConstructor,
    LSTMTrainer,
)
from utils import setup_logger, ensure_dir, save_pickle, save_json
from utils.match_calibration import estimate_unknown_threshold_from_clusters


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r.returncode == 0 and (r.stdout or "").strip():
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return ""


def apply_json_config_overrides(cfg: RSTPMConfig, updates: dict) -> None:
    """将 JSON 递归写入 RSTPMConfig 及嵌套 dataclass（仅覆盖存在的字段）。"""

    def walk(dc, d: dict) -> None:
        for k, v in d.items():
            if not hasattr(dc, k):
                continue
            cur = getattr(dc, k)
            if isinstance(v, dict) and is_dataclass(cur):
                walk(cur, v)
            else:
                setattr(dc, k, v)

    walk(cfg, updates)


def _persist_step_dataframe(
    cfg: RSTPMConfig,
    logger,
    step_num: int,
    out_dir: str,
    df,
    basename: str,
) -> None:
    if not cfg.paths.persist_intermediate_tables:
        return
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{basename}.parquet")
    df.to_parquet(path, index=False)
    save_json(
        {
            "step": step_num,
            "schema_version": "1",
            "rows": int(len(df)),
            "parquet": os.path.basename(path),
            "saved_at_unix": time.time(),
        },
        os.path.join(out_dir, f"step{step_num}_complete.json"),
    )
    logger.info("  步骤%s 中间落盘: %s 行 → %s", step_num, len(df), path)


def _persist_step_trajectories(
    cfg: RSTPMConfig,
    logger,
    step_num: int,
    out_dir: str,
    trajectories,
    basename: str,
) -> None:
    if not cfg.paths.persist_intermediate_tables:
        return
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{basename}.pkl")
    save_pickle(trajectories, path)
    total = sum(len(t) for t in trajectories) if trajectories else 0
    save_json(
        {
            "step": step_num,
            "schema_version": "1",
            "traj_count": len(trajectories),
            "total_rows": int(total),
            "pickle": os.path.basename(path),
            "saved_at_unix": time.time(),
        },
        os.path.join(out_dir, f"step{step_num}_complete.json"),
    )
    logger.info(
        "  步骤%s 中间落盘: %s 条轨迹 → %s", step_num, len(trajectories), path
    )


def _build_run_config_snapshot(cfg: RSTPMConfig) -> dict:
    """清单 45：可 JSON 序列化的运行参数摘要（避免 tuple 键等不可序列化对象）。"""
    return {
        "schema_version": "1",
        "study_name": cfg.domain_focus.study_name,
        "roi_version": cfg.domain_focus.roi_version,
        "filter_profile": cfg.domain_focus.filter_profile,
        "domain_focus_enabled": cfg.domain_focus.enabled,
        "domain_focus_spatial_active": domain_focus_spatial_active(cfg.domain_focus),
        "sog_max_knots": cfg.port_cleaning.sog_max_knots,
        "gap_hours_for_split": cfg.port_cleaning.gap_hours_for_split,
        "resample_interval_sec": cfg.resample.resample_interval,
        "resample_heading_mode": cfg.resample.resample_heading_mode,
        "resample_output_subdir": cfg.resample.output_subdir,
        "berth_mode": cfg.berth.mode,
        "allow_missing_cog_rows": cfg.filter.allow_missing_cog_rows,
        "csv_chunksize": cfg.data_load.csv_chunksize,
    }


def run_offline_pipeline(config: RSTPMConfig, step1_incremental: bool = False):
    """
    执行完整的离线训练流水线。
    
    Parameters
    ----------
    config : RSTPMConfig — 全局配置
    """
    logger = setup_logger("OfflinePipeline", config.paths.log_dir)

    sync_port_cleaning_to_pipeline(config)
    apply_domain_roi_to_filter(config)
    if not config.resample_experiment.run_all_ablation_intervals:
        apply_resample_experiment_interval(config)

    save_json(
        _build_run_config_snapshot(config),
        os.path.join(config.paths.log_dir, "run_config_snapshot.json"),
    )
    
    # ===== 创建输出目录 =====
    for d in [config.paths.intermediate_dir, config.paths.cleaned_data_dir,
              config.paths.no_berth_data_dir, config.paths.trajectories_dir,
              config.paths.resampled_dir, config.paths.cluster_dir,
              config.paths.normalization_dir, config.paths.model_dir,
              config.paths.log_dir, config.paths.figure_dir,
              config.paths.subsets_raw_daily_dir, config.paths.subsets_metadata_dir,
              config.paths.subsets_merged_monthly_dir]:
        ensure_dir(d)
    
    total_start = time.time()
    
    # ═════════════════════════════════════════════════════════════════════════
    # 步骤1：加载原始AIS数据
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("步骤1：加载原始AIS数据...")
    step_start = time.time()
    
    loader = AISDataLoader(
        config.paths,
        config.data_load,
        domain_config=config.domain_focus,
        port_cleaning=config.port_cleaning,
    )
    if step1_incremental:
        raw_df = loader.run_with_checkpoint()
    else:
        raw_df = loader.run()
    
    logger.info(f"  步骤1完成: {len(raw_df)} 条记录, "
                f"耗时 {time.time() - step_start:.1f}s")
    logger.info(
        "  轨迹连续性说明：步骤4 切分基于全期合并表按 MMSI+Timestamp(UTC) 排序，"
        "不按自然日独立切分（清单 21）。"
    )

    if domain_focus_spatial_active(config.domain_focus):
        logger.info("=" * 60)
        logger.info("步骤1b：业务域空间精筛（polygon / corridor / exclusion）...")
        t1b = time.time()
        raw_df = DomainSpatialFilter(config.domain_focus).run(raw_df)
        logger.info(
            "  步骤1b完成: %s 条记录, 耗时 %.1fs",
            len(raw_df),
            time.time() - t1b,
        )
    
    # ═════════════════════════════════════════════════════════════════════════
    # 步骤2：异常值过滤
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("步骤2：异常值检测与过滤...")
    step_start = time.time()
    
    anomaly_filter = AnomalyFilter(config.filter, port_cleaning=config.port_cleaning)
    clean_df = anomaly_filter.run(raw_df)
    
    logger.info(f"  步骤2完成: {len(raw_df)} → {len(clean_df)} 条记录, "
                f"耗时 {time.time() - step_start:.1f}s")
    logger.info(f"  过滤统计: {anomaly_filter.get_stats()}")
    _persist_step_dataframe(
        config, logger, 2, config.paths.cleaned_data_dir, clean_df, "step2_cleaned"
    )
    del raw_df  # 释放内存
    
    # ═════════════════════════════════════════════════════════════════════════
    # 步骤3：停泊段检测与删除
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("步骤3：停泊段检测与删除...")
    step_start = time.time()
    
    berth_detector = BerthDetector(config.berth)
    no_berth_df = berth_detector.run(clean_df)
    
    logger.info(f"  步骤3完成: {len(clean_df)} → {len(no_berth_df)} 条记录, "
                f"耗时 {time.time() - step_start:.1f}s")
    _persist_step_dataframe(
        config,
        logger,
        3,
        config.paths.no_berth_data_dir,
        no_berth_df,
        "step3_no_berth",
    )
    del clean_df
    
    # ═════════════════════════════════════════════════════════════════════════
    # 步骤4：轨迹切分
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("步骤4：航行轨迹切分...")
    step_start = time.time()
    
    splitter = TrajectorySplitter(
        config.traj_split, domain_focus=config.domain_focus
    )
    trajectories = splitter.run(no_berth_df)
    
    logger.info(f"  步骤4完成: {len(trajectories)} 条独立轨迹, "
                f"耗时 {time.time() - step_start:.1f}s")
    _persist_step_trajectories(
        config,
        logger,
        4,
        config.paths.trajectories_dir,
        trajectories,
        "step4_trajectories",
    )
    del no_berth_df
    
    # ═════════════════════════════════════════════════════════════════════════
    # 步骤5：线性插值重采样
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("步骤5：线性插值与等时间间隔重采样...")
    step_start = time.time()

    if config.resample_experiment.run_all_ablation_intervals:
        intervals = sorted(
            {float(x) for x in config.resample_experiment.ablation_intervals_sec}
        )
        if not intervals:
            intervals = [float(config.resample_experiment.baseline_interval_sec)]
    else:
        intervals = [float(config.resample.resample_interval)]

    baseline_sec = float(config.resample_experiment.baseline_interval_sec)
    resampled_for_downstream = None
    last_one = None
    downstream_iv = baseline_sec
    for iv in intervals:
        config.resample.resample_interval = iv
        config.resample.output_subdir = f"dt{int(iv)}"
        resampler = TrajectoryResampler(config.resample)
        resampled_one = resampler.run(trajectories)
        last_one = resampled_one
        res_dir = resampler.resolved_output_dir(config.paths.resampled_dir)
        ensure_dir(res_dir)
        save_pickle(
            resampled_one,
            os.path.join(res_dir, "resampled_trajectories.pkl"),
        )
        if abs(iv - baseline_sec) < 1e-6:
            resampled_for_downstream = resampled_one
            downstream_iv = iv
    if resampled_for_downstream is None:
        resampled_for_downstream = last_one
        downstream_iv = intervals[-1]
    resampled_trajectories = resampled_for_downstream
    config.resample.resample_interval = float(downstream_iv)
    config.resample.output_subdir = f"dt{int(downstream_iv)}"

    total_points_before = sum(len(t) for t in trajectories)
    total_points_after = sum(len(t) for t in resampled_trajectories)
    logger.info(f"  步骤5完成: {len(resampled_trajectories)} 条轨迹, "
                f"数据点 {total_points_before} → {total_points_after}, "
                f"耗时 {time.time() - step_start:.1f}s")
    
    # 保存中间结果（大量计算后保底存盘；与下游聚类使用同一列表）
    _main_resampler = TrajectoryResampler(config.resample)
    _main_res_dir = _main_resampler.resolved_output_dir(config.paths.resampled_dir)
    ensure_dir(_main_res_dir)
    save_pickle(
        resampled_trajectories,
        os.path.join(_main_res_dir, "resampled_trajectories.pkl"),
    )
    
    # ═════════════════════════════════════════════════════════════════════════
    # 步骤6：CDDTW聚类
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("步骤6：CDDTW聚类（最复杂步骤）...")
    step_start = time.time()
    
    # 6A: 提取起终点
    logger.info("  6A: 提取起终点...")
    endpoint_extractor = EndpointExtractor()
    start_points, end_points, traj_indices = endpoint_extractor.extract(
        resampled_trajectories)
    
    # 6B: DBSCAN聚类起终点 → 初始分组
    logger.info("  6B: DBSCAN聚类起终点...")
    se_clusterer = StartEndClusterer(config.cluster)
    initial_groups = se_clusterer.run(start_points, end_points, traj_indices)
    logger.info(f"    初始分组数: {len(initial_groups)}")
    
    # 6C: DTW细化
    logger.info("  6C: DTW轨迹相似度细化...")
    refiner = TrajectoryRefiner(config.cluster)
    refined_clusters = refiner.run(initial_groups, resampled_trajectories)
    logger.info(f"    细化后类别数: {len(refined_clusters)}")
    
    # 6D+6E: 跨类别合并 + 最终输出
    logger.info("  6D: 跨类别合并...")
    merger = ClusterMerger(config.cluster)
    final_clusters = merger.run(refined_clusters, resampled_trajectories)
    
    logger.info(f"  步骤6完成: {len(final_clusters)} 个最终聚类类别, "
                f"耗时 {time.time() - step_start:.1f}s")
    
    # 保存聚类结果
    save_pickle(final_clusters,
                os.path.join(config.paths.cluster_dir, "final_clusters.pkl"))

    # 方案 6.5：聚类清单（训练完成后会再次写入完整路径）
    manifest_partial = {
        "schema_version": "1",
        "stage": "after_clustering",
        "clusters": [
            {
                "cluster_id": int(c["cluster_id"]),
                "traj_count": len(c["traj_indices"]),
                "rep_traj_idx": int(c["rep_traj_idx"]),
                "intra_avg_dist": float(c.get("intra_avg_dist", 0.0)),
            }
            for c in final_clusters
        ],
    }
    save_json(
        manifest_partial,
        os.path.join(config.paths.cluster_dir, "cluster_manifest.json"),
    )

    # 方案 11.3：离线交叉特征匹配距离 → 建议 unknown 阈值（在线可加载）
    _sug, cal_meta = estimate_unknown_threshold_from_clusters(
        final_clusters,
        calibration_T=config.train.default_time_steps,
    )
    save_json(
        cal_meta,
        os.path.join(config.paths.cluster_dir, "match_unknown_threshold.json"),
    )
    logger.info(
        "match_unknown_threshold.json 已写入 suggested=%s (T=%s)",
        cal_meta.get("unknown_threshold_suggested", _sug),
        cal_meta.get("calibration_T"),
    )

    # 方案 11.4：MMSI 在各 cluster 上的历史航次条数 N_k（在线分叉概率）
    mmsi_map: dict = {}
    for c in final_clusters:
        cid = int(c["cluster_id"])
        for tid in c["traj_indices"]:
            m = int(resampled_trajectories[tid]["MMSI"].iloc[0])
            if m not in mmsi_map:
                mmsi_map[m] = {}
            mmsi_map[m][cid] = mmsi_map[m].get(cid, 0) + 1
    mmsi_cluster_payload = {
        "schema_version": "1",
        "mmsi_clusters": {
            str(m): {str(k): v for k, v in inner.items()}
            for m, inner in mmsi_map.items()
        },
    }
    save_json(
        mmsi_cluster_payload,
        os.path.join(config.paths.cluster_dir, "mmsi_cluster_counts.json"),
    )
    logger.info(
        "mmsi_cluster_counts.json 已写入: %s 艘 MMSI",
        len(mmsi_map),
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 步骤7 + 8 + 9：对每个聚类类别 → 归一化 → 构造样本 → 训练LSTM
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("步骤7~9：对每个聚类类别训练LSTM模型...")
    step_start = time.time()
    
    normalizer = DataNormalizer(config.normalization, config.paths)
    sample_constructor = SampleConstructor(config.train, config.split)
    trainer = LSTMTrainer(config.train, config.paths, config.device)
    
    training_results = {}
    
    for cluster_info in final_clusters:
        cid = cluster_info["cluster_id"]
        logger.info(f"  --- 类别 {cid} ---")
        logger.info(f"    轨迹数: {len(cluster_info['traj_indices'])}")
        
        # 步骤7: 归一化
        normalized_trajs, norm_params = normalizer.normalize_cluster(
            cluster_info, resampled_trajectories)
        norm_params.save(os.path.join(
            config.paths.normalization_dir, f"norm_params_cluster_{cid}.json"))
        
        # 步骤8: 构造样本
        sample_info = sample_constructor.run(
            cluster_info, normalized_trajs, resampled_trajectories)
        T, H = sample_info["T"], sample_info["H"]
        logger.info(f"    T={T}, H={H}")
        
        # 步骤9: 训练LSTM
        result = trainer.train_cluster_model(
            cluster_id=cid,
            T=T, H=H,
            train_loader=sample_info["train_loader"],
            val_loader=sample_info["val_loader"],
            test_loader=sample_info["test_loader"],
        )
        
        training_results[cid] = result
        logger.info(f"    训练完成: val_loss={result['best_val_loss']:.6f}, "
                    f"test_loss={result['test_loss']:.6f}, "
                    f"test_acc={result['test_accuracy']:.4f}")

        # 方案 9.9：特征轨迹 npz + ModelBundleDescriptor
        feat_npz = os.path.join(
            config.paths.cluster_dir, f"feature_traj_cluster_{cid}.npz"
        )
        np.savez_compressed(
            feat_npz,
            lon=np.asarray(cluster_info["rep_traj_lon"], dtype=np.float64),
            lat=np.asarray(cluster_info["rep_traj_lat"], dtype=np.float64),
        )
        norm_abs = os.path.abspath(
            os.path.join(
                config.paths.normalization_dir, f"norm_params_cluster_{cid}.json"
            )
        )
        weights_abs = os.path.abspath(result["model_path"])
        gh = _git_head()
        bundle = ModelBundleDescriptor(
            cluster_id=cid,
            T=T,
            H=H,
            input_size=config.train.input_features,
            output_size=config.train.output_features,
            weights_path=weights_abs,
            norm_params_path=norm_abs,
            rep_traj_path=os.path.abspath(feat_npz),
            git_commit=gh if gh else None,
        )
        bd = asdict(bundle)
        bd["feature_order"] = list(bundle.feature_order)
        save_json(
            bd,
            os.path.join(
                config.paths.model_dir, f"model_bundle_cluster_{cid}.json"
            ),
        )

    manifest_final = {
        "schema_version": "1",
        "stage": "after_training",
        # 与在线阶段步骤 11 衔接的聚类目录辅助产物（本轮审计闭环）
        "auxiliary_artifacts": {
            "match_unknown_threshold_json": "match_unknown_threshold.json",
            "mmsi_cluster_counts_json": "mmsi_cluster_counts.json",
        },
        "clusters": [],
    }
    for c in final_clusters:
        cid = int(c["cluster_id"])
        manifest_final["clusters"].append(
            {
                "cluster_id": cid,
                "traj_count": len(c["traj_indices"]),
                "rep_traj_idx": int(c["rep_traj_idx"]),
                "intra_avg_dist": float(c.get("intra_avg_dist", 0.0)),
                "feature_traj_npz": f"feature_traj_cluster_{cid}.npz",
                "model_bundle_json": f"model_bundle_cluster_{cid}.json",
                "weights_file": f"lstm_cluster_{cid}.pt",
                "norm_params_json": f"norm_params_cluster_{cid}.json",
            }
        )
    save_json(
        manifest_final,
        os.path.join(config.paths.cluster_dir, "cluster_manifest.json"),
    )

    logger.info(f"步骤7~9全部完成, 耗时 {time.time() - step_start:.1f}s")
    
    # ═════════════════════════════════════════════════════════════════════════
    # 完成
    # ═════════════════════════════════════════════════════════════════════════
    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"离线训练全链路完成! 总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"模型保存位置: {config.paths.model_dir}")
    logger.info(f"聚类结果位置: {config.paths.cluster_dir}")
    
    return training_results


def main():
    parser = argparse.ArgumentParser(description="RSTPM 离线训练流水线")
    parser.add_argument("--raw_data_dir", type=str, default=None,
                        help="原始AIS数据zip目录")
    parser.add_argument("--date_start", type=str, default=None,
                        help="起始日期 YYYY-MM-DD")
    parser.add_argument("--date_end", type=str, default=None,
                        help="结束日期 YYYY-MM-DD")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="训练设备")
    parser.add_argument(
        "--step1_incremental",
        action="store_true",
        help="步骤1：按日写 parquet 并更新 pipeline_checkpoint（大规模数据）",
    )
    parser.add_argument(
        "--enable_domain",
        action="store_true",
        help="启用 DomainFocus（LA/LB bbox、子集布局、步骤1b 空间精筛）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 覆盖 RSTPMConfig（嵌套键与 dataclass 字段同名）",
    )
    args = parser.parse_args()
    
    config = get_default_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            apply_json_config_overrides(config, json.load(f))
        sync_port_cleaning_to_pipeline(config)
        apply_domain_roi_to_filter(config)
        if not config.resample_experiment.run_all_ablation_intervals:
            apply_resample_experiment_interval(config)
    
    # 覆盖命令行参数
    if args.raw_data_dir:
        config.paths.raw_data_dir = args.raw_data_dir
    if args.date_start:
        config.data_load.date_range_start = args.date_start
    if args.date_end:
        config.data_load.date_range_end = args.date_end
    config.device = args.device
    if args.enable_domain:
        config.domain_focus.enabled = True

    run_offline_pipeline(config, step1_incremental=args.step1_incremental)


if __name__ == "__main__":
    main()
