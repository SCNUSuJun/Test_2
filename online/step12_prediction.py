"""
步骤12：在线阶段——执行LSTM模型推理并输出预测轨迹
==================================================
"""

import torch
import numpy as np
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

from config import PredictConfig, ResampleConfig
from schemas.assets import OnlinePredictionResult
from model.step7_normalization import NormalizationParams
from model.step9_lstm_train import LSTMPredictor
from online.step10_realtime_preprocess import VesselBuffer
from utils import setup_logger


class TrajectoryPredictor:
    """LSTM 多步递推预测。"""

    def __init__(
        self,
        predict_config: PredictConfig,
        resample_config: ResampleConfig,
        device: str = "cpu",
    ):
        self.predict_cfg = predict_config
        self.resample_cfg = resample_config
        self.device = torch.device(device)
        self.logger = setup_logger("TrajectoryPredictor")

    def normalize_input(
        self, raw_sequence: np.ndarray, params: NormalizationParams
    ) -> np.ndarray:
        eps = 1e-8
        out = np.empty_like(raw_sequence, dtype=np.float64)
        cols = [
            (0, params.lon_min, params.lon_max),
            (1, params.lat_min, params.lat_max),
            (2, params.sog_min, params.sog_max),
            (3, params.cog_min, params.cog_max),
        ]
        for j, lo, hi in cols:
            denom = hi - lo
            if abs(denom) < eps:
                out[:, j] = 0.5
            else:
                out[:, j] = (raw_sequence[:, j] - lo) / (denom + eps)
        return out

    def predict_single_step(
        self, model: LSTMPredictor, input_sequence: np.ndarray
    ) -> np.ndarray:
        model.eval()
        x = torch.from_numpy(input_sequence.astype(np.float32)).unsqueeze(0).to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            y = model(x)
        return y.squeeze(0).cpu().numpy()

    def denormalize_output(
        self,
        normalized_pred: np.ndarray,
        params: NormalizationParams,
        epsilon: float = 1e-8,
    ) -> Tuple[float, float, float, float]:
        def inv(x, lo, hi):
            d = hi - lo
            if abs(d) < epsilon:
                return lo
            return float(x * d + lo)

        return (
            inv(normalized_pred[0], params.lon_min, params.lon_max),
            inv(normalized_pred[1], params.lat_min, params.lat_max),
            inv(normalized_pred[2], params.sog_min, params.sog_max),
            inv(normalized_pred[3], params.cog_min, params.cog_max),
        )

    def predict_multi_step(
        self,
        model: LSTMPredictor,
        vessel_buffer: VesselBuffer,
        params: NormalizationParams,
        T: int,
        num_steps: int,
        current_timestamp: float,
    ) -> List[Dict]:
        raw = vessel_buffer.get_sequence(T)
        if raw.shape[0] < T:
            return []
        dt = self.resample_cfg.resample_interval
        window_norm = self.normalize_input(raw, params)
        preds: List[Dict] = []
        dev = next(model.parameters()).device
        model.eval()
        for k in range(num_steps):
            x = torch.from_numpy(window_norm.astype(np.float32)).unsqueeze(0).to(dev)
            with torch.no_grad():
                yhat = model(x).squeeze(0).cpu().numpy()
            lon, lat, sog, cog = self.denormalize_output(yhat, params)
            t_pred = current_timestamp + (k + 1) * dt
            preds.append(
                {
                    "step": k + 1,
                    "timestamp": t_pred,
                    "lon": lon,
                    "lat": lat,
                    "sog": sog,
                    "cog": cog,
                }
            )
            window_norm = np.vstack([window_norm[1:], yhat.reshape(1, -1)])
        return preds

    def predict_with_fork(
        self,
        models_info: List[Dict],
        vessel_buffer: VesselBuffer,
        current_timestamp: float,
    ) -> Dict[int, List[Dict]]:
        out: Dict[int, List[Dict]] = {}
        for info in models_info:
            cid = info["cluster_id"]
            P = self.predict_cfg.prediction_steps
            out[cid] = self.predict_multi_step(
                info["model"],
                vessel_buffer,
                info["norm_params"],
                info["T"],
                P,
                current_timestamp,
            )
        return out

    @staticmethod
    def _normalize_step_dict(p: Dict) -> Dict[str, float]:
        """统一为契约字段 lon,lat,sog,cog,t（方案 12.7）。"""
        ts = p.get("timestamp", p.get("t"))
        if ts is None:
            raise ValueError(
                "预测步字典须含 timestamp 或 t 字段（OnlinePredictionResult / 方案 12.7）"
            )
        return {
            "lon": float(p["lon"]),
            "lat": float(p["lat"]),
            "sog": float(p["sog"]),
            "cog": float(p["cog"]),
            "t": float(ts),
        }

    def format_prediction_output(
        self,
        mmsi: int,
        cluster_id: int,
        predictions: List[Dict],
        fork_probabilities: Optional[Dict] = None,
        is_fork: bool = False,
        branch_predictions: Optional[Dict[int, List[Dict]]] = None,
    ) -> Dict:
        points = [self._normalize_step_dict(p) for p in predictions]
        br: Optional[Dict[int, List[Dict[str, float]]]] = None
        if branch_predictions is not None:
            br = {
                int(cid): [self._normalize_step_dict(x) for x in seq]
                for cid, seq in branch_predictions.items()
            }
        bundle = OnlinePredictionResult(
            mmsi=int(mmsi),
            cluster_id=int(cluster_id) if cluster_id is not None else None,
            points=points,
            prediction_count=len(points),
            is_fork=is_fork,
            fork_probabilities=(
                {int(k): float(v) for k, v in fork_probabilities.items()}
                if fork_probabilities
                else None
            ),
            branch_predictions=br,
            schema_version="1",
        )
        return asdict(bundle)
