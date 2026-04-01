"""
步骤9：为每个聚类类别训练一个LSTM预测模型
==========================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any
import numpy as np
import json

from config import TrainConfig, PathConfig
from utils import setup_logger, ensure_dir


class LSTMPredictor(nn.Module):
    """Input (B,T,4) → LSTM → Dropout → Linear → (B,4)"""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        dropout: float = 0.2,
        output_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


class LSTMTrainer:
    """LSTM 训练、早停、保存。"""

    def __init__(
        self, train_config: TrainConfig, path_config: PathConfig, device: str = "cuda"
    ):
        self.train_cfg = train_config
        self.path_cfg = path_config
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.logger = setup_logger("LSTMTrainer")

    def build_model(self, hidden_size: int) -> LSTMPredictor:
        m = LSTMPredictor(
            input_size=self.train_cfg.input_features,
            hidden_size=hidden_size,
            dropout=self.train_cfg.dropout,
            output_size=self.train_cfg.output_features,
        )
        return m.to(self.device)

    def train_one_epoch(
        self,
        model: LSTMPredictor,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        return total / max(n, 1)

    def evaluate(
        self,
        model: LSTMPredictor,
        data_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        model.eval()
        total, n = 0.0, 0
        sse = 0.0
        feat = self.train_cfg.output_features
        with torch.no_grad():
            for xb, yb in data_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = model(xb)
                loss = criterion(pred, yb)
                total += loss.item() * xb.size(0)
                n += xb.size(0)
                sse += torch.sum((pred - yb) ** 2).item()
        avg_loss = total / max(n, 1)
        mse = sse / max(n * feat, 1) if n else 0.0
        # 归一化空间 1/(1+MSE)，非地理精度；日志字段仍沿用 *_acc 以免破坏既有解析
        norm_fit = float(1.0 / (1.0 + mse))
        return avg_loss, norm_fit

    def train_cluster_model(
        self,
        cluster_id: int,
        T: int,
        H: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Dict[str, Any]:
        ensure_dir(self.path_cfg.model_dir)
        model = self.build_model(H)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.train_cfg.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.train_cfg.lr_scheduler_factor,
            patience=self.train_cfg.lr_scheduler_patience,
        )
        best_val = float("inf")
        best_state = None
        patience_left = self.train_cfg.early_stop_patience
        log: List[Dict] = []

        if len(train_loader) == 0:
            path = os.path.join(self.path_cfg.model_dir, f"lstm_cluster_{cluster_id}.pt")
            self.save_model(model, cluster_id, T, H, path)
            return {
                "model_path": path,
                "best_val_loss": float("nan"),
                "test_loss": float("nan"),
                "test_accuracy": 0.0,
                "training_log": [],
                "epochs_trained": 0,
            }

        for epoch in range(self.train_cfg.epochs):
            tr_loss = self.train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = self.evaluate(model, val_loader, criterion)
            te_loss, te_acc = self.evaluate(model, test_loader, criterion)
            scheduler.step(val_loss)
            log.append(
                {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_loss": te_loss,
                    "test_acc": te_acc,
                }
            )
            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = self.train_cfg.early_stop_patience
            else:
                patience_left -= 1
            if patience_left <= 0:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        path = os.path.join(self.path_cfg.model_dir, f"lstm_cluster_{cluster_id}.pt")
        self.save_model(model, cluster_id, T, H, path)
        self.save_training_log(cluster_id, log)
        test_loss, test_acc = self.evaluate(model, test_loader, criterion)
        return {
            "model_path": path,
            "best_val_loss": float(best_val) if best_state is not None else float("nan"),
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "training_log": log,
            "epochs_trained": len(log),
        }

    def save_model(
        self, model: LSTMPredictor, cluster_id: int, T: int, H: int, model_path: str
    ) -> None:
        ensure_dir(os.path.dirname(os.path.abspath(model_path)) or ".")
        payload = {
            "state_dict": model.state_dict(),
            "cluster_id": cluster_id,
            "T": T,
            "H": H,
            "input_size": self.train_cfg.input_features,
            "output_size": self.train_cfg.output_features,
            "dropout": self.train_cfg.dropout,
        }
        torch.save(payload, model_path)

    def load_model(self, model_path: str) -> Tuple[LSTMPredictor, Dict]:
        try:
            payload = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            payload = torch.load(model_path, map_location=self.device)
        H = int(payload["H"])
        model = LSTMPredictor(
            input_size=int(payload.get("input_size", 4)),
            hidden_size=H,
            dropout=float(payload.get("dropout", 0.2)),
            output_size=int(payload.get("output_size", 4)),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(self.device)
        cfg = {
            "cluster_id": payload["cluster_id"],
            "T": payload["T"],
            "H": payload["H"],
        }
        return model, cfg

    def save_training_log(self, cluster_id: int, log: List[Dict]) -> None:
        p = os.path.join(self.path_cfg.log_dir, f"train_cluster_{cluster_id}.json")
        ensure_dir(self.path_cfg.log_dir)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
