"""
评估指标模块
============

对应方案 附录B 的精度基准。
"""

import numpy as np
from typing import List, Dict, Tuple, Any

from utils import haversine_distance_vectorized


def _heading_error_vectorized(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    d = np.abs(pred.astype(np.float64) - true.astype(np.float64)) % 360.0
    return np.minimum(d, 360.0 - d)


class PredictionEvaluator:
    """对比预测轨迹与真值，计算附录 B 相关指标。"""

    def __init__(self):
        pass

    def compute_distance_error(
        self,
        pred_lon: np.ndarray,
        pred_lat: np.ndarray,
        true_lon: np.ndarray,
        true_lat: np.ndarray,
    ) -> np.ndarray:
        if not (len(pred_lon) == len(pred_lat) == len(true_lon) == len(true_lat)):
            raise ValueError("经纬度序列长度必须一致")
        return haversine_distance_vectorized(
            pred_lat, pred_lon, true_lat, true_lon
        )

    def compute_speed_error(
        self, pred_sog: np.ndarray, true_sog: np.ndarray
    ) -> np.ndarray:
        if len(pred_sog) != len(true_sog):
            raise ValueError("航速序列长度必须一致")
        return np.abs(pred_sog.astype(np.float64) - true_sog.astype(np.float64))

    def compute_heading_error(
        self, pred_cog: np.ndarray, true_cog: np.ndarray
    ) -> np.ndarray:
        if len(pred_cog) != len(true_cog):
            raise ValueError("航向序列长度必须一致")
        return _heading_error_vectorized(pred_cog, true_cog)

    def compute_norm_space_fit(self, pred: np.ndarray, true: np.ndarray) -> float:
        """
        归一化特征空间的拟合度 1/(1+MSE)，与 LSTMTrainer.evaluate 中 acc 定义一致。
        不是地理米级精度或业务「准确率」；对外汇报应优先 ADE/MDE 与速度/航向误差。
        """
        p = np.asarray(pred, dtype=np.float64).ravel()
        t = np.asarray(true, dtype=np.float64).ravel()
        if p.size != t.size:
            raise ValueError("pred 与 true 元素数必须一致")
        if p.size == 0:
            return 0.0
        mse = float(np.mean((p - t) ** 2))
        return float(1.0 / (1.0 + mse))

    def compute_accuracy(self, pred: np.ndarray, true: np.ndarray) -> float:
        """兼容旧名；请优先使用 compute_norm_space_fit。"""
        return self.compute_norm_space_fit(pred, true)

    def _lists_to_arrays(
        self, predictions: List[Dict], ground_truth: List[Dict]
    ) -> Tuple[np.ndarray, ...]:
        n = min(len(predictions), len(ground_truth))
        if n == 0:
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
            )

        def get_key(row: Dict, *keys: str) -> float:
            for k in keys:
                if k in row:
                    return float(row[k])
            raise KeyError(f"缺少字段: {keys}")

        plon, plat, psog, pcog = [], [], [], []
        tlon, tlat, tsog, tcog = [], [], [], []
        for i in range(n):
            pr, tr = predictions[i], ground_truth[i]
            plon.append(get_key(pr, "lon", "LON"))
            plat.append(get_key(pr, "lat", "LAT"))
            psog.append(get_key(pr, "sog", "SOG"))
            pcog.append(get_key(pr, "cog", "COG"))
            tlon.append(get_key(tr, "lon", "LON"))
            tlat.append(get_key(tr, "lat", "LAT"))
            tsog.append(get_key(tr, "sog", "SOG"))
            tcog.append(get_key(tr, "cog", "COG"))
        return (
            np.array(plon),
            np.array(plat),
            np.array(psog),
            np.array(pcog),
            np.array(tlon),
            np.array(tlat),
            np.array(tsog),
            np.array(tcog),
        )

    def evaluate_single_trajectory(
        self, predictions: List[Dict], ground_truth: List[Dict]
    ) -> Dict[str, Any]:
        (
            plon,
            plat,
            psog,
            pcog,
            tlon,
            tlat,
            tsog,
            tcog,
        ) = self._lists_to_arrays(predictions, ground_truth)
        if len(plon) == 0:
            return {
                "ade": float("nan"),
                "mde": float("nan"),
                "avg_speed_error": float("nan"),
                "max_speed_error": float("nan"),
                "avg_heading_error": float("nan"),
                "max_heading_error": float("nan"),
                "per_step_distance": np.array([]),
                "norm_space_fit": float("nan"),
                "norm_accuracy": float("nan"),
            }
        dist = self.compute_distance_error(plon, plat, tlon, tlat)
        spd = self.compute_speed_error(psog, tsog)
        hdg = self.compute_heading_error(pcog, tcog)
        pred_nf = np.column_stack([plon, plat, psog, pcog])
        true_nf = np.column_stack([tlon, tlat, tsog, tcog])
        return {
            "ade": float(np.mean(dist)),
            "mde": float(np.max(dist)),
            "avg_speed_error": float(np.mean(spd)),
            "max_speed_error": float(np.max(spd)),
            "avg_heading_error": float(np.mean(hdg)),
            "max_heading_error": float(np.max(hdg)),
            "per_step_distance": dist,
            "norm_space_fit": self.compute_norm_space_fit(pred_nf, true_nf),
            "norm_accuracy": self.compute_norm_space_fit(pred_nf, true_nf),
        }

    def evaluate_cluster_model(
        self,
        cluster_id: int,
        all_predictions: List[List[Dict]],
        all_ground_truths: List[List[Dict]],
    ) -> Dict[str, Any]:
        if len(all_predictions) != len(all_ground_truths):
            raise ValueError("预测与真值轨迹条数不一致")
        ades, mdes = [], []
        avs, mxs = [], []
        avh, mxh = [], []
        naccs = []
        for pred, gt in zip(all_predictions, all_ground_truths):
            m = self.evaluate_single_trajectory(pred, gt)
            if len(m["per_step_distance"]) == 0:
                continue
            ades.append(m["ade"])
            mdes.append(m["mde"])
            avs.append(m["avg_speed_error"])
            mxs.append(m["max_speed_error"])
            avh.append(m["avg_heading_error"])
            mxh.append(m["max_heading_error"])
            naccs.append(m["norm_space_fit"])
        if not ades:
            return {
                "cluster_id": cluster_id,
                "n_trajectories": 0,
                "ade_mean": float("nan"),
                "ade_median": float("nan"),
                "ade_std": float("nan"),
                "mde_mean": float("nan"),
                "mde_max": float("nan"),
            }
        return {
            "cluster_id": cluster_id,
            "n_trajectories": len(ades),
            "ade_mean": float(np.mean(ades)),
            "ade_median": float(np.median(ades)),
            "ade_std": float(np.std(ades)),
            "mde_mean": float(np.mean(mdes)),
            "mde_max": float(np.max(mdes)),
            "avg_speed_error_mean": float(np.mean(avs)),
            "max_speed_error_mean": float(np.mean(mxs)),
            "avg_heading_error_mean": float(np.mean(avh)),
            "max_heading_error_mean": float(np.mean(mxh)),
            "norm_space_fit_mean": float(np.mean(naccs)),
            "norm_accuracy_mean": float(np.mean(naccs)),
        }

    def generate_report(self, results: Dict[int, Dict]) -> str:
        lines = ["# RSTPM 预测评估报告（附录 B 对照）", ""]
        for cid in sorted(results.keys()):
            r = results[cid]
            lines.append(f"## 聚类类别 {cid}")
            lines.append(f"- 轨迹数: {r.get('n_trajectories', 0)}")
            for k in (
                "ade_mean",
                "ade_median",
                "ade_std",
                "mde_mean",
                "mde_max",
                "avg_speed_error_mean",
                "max_speed_error_mean",
                "avg_heading_error_mean",
                "max_heading_error_mean",
                "norm_space_fit_mean",
                "norm_accuracy_mean",
            ):
                if k in r:
                    lines.append(f"- {k}: {r[k]:.6g}" if isinstance(r[k], float) else f"- {k}: {r[k]}")
            lines.append("")
        return "\n".join(lines)
