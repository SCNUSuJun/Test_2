"""
步骤6B：使用Ball-Tree优化的DBSCAN分别对起点和终点进行空间聚类
=============================================================
"""

import numpy as np
from typing import Dict, Tuple, List

from sklearn.cluster import DBSCAN

from config import ClusterConfig
from utils import setup_logger


class StartEndClusterer:
    """起终点空间聚类器（sklearn Ball_tree + DBSCAN）。"""

    def __init__(self, config: ClusterConfig):
        self.cfg = config
        self.logger = setup_logger("StartEndClusterer")

    def build_ball_tree(self, points: np.ndarray):
        from sklearn.neighbors import BallTree

        if len(points) == 0:
            return None
        return BallTree(points, leaf_size=self.cfg.ball_tree_leaf_size, metric="euclidean")

    def dbscan_with_balltree(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.array([], dtype=int)
        clf = DBSCAN(
            eps=self.cfg.dbscan_eps,
            min_samples=self.cfg.dbscan_min_samples,
            metric="euclidean",
            algorithm="ball_tree",
            leaf_size=self.cfg.ball_tree_leaf_size,
        )
        return clf.fit_predict(points)

    def form_initial_groups(
        self,
        start_labels: np.ndarray,
        end_labels: np.ndarray,
        traj_indices: List[int],
    ) -> Dict[Tuple[int, int], List[int]]:
        groups: Dict[Tuple[int, int], List[int]] = {}
        for k, (s, e) in enumerate(zip(start_labels, end_labels)):
            tid = traj_indices[k]
            if s == -1 or e == -1:
                key = (-1, -1)
            else:
                key = (int(s), int(e))
            groups.setdefault(key, []).append(tid)
        return groups

    def run(
        self,
        start_points: np.ndarray,
        end_points: np.ndarray,
        traj_indices: List[int],
    ) -> Dict[Tuple[int, int], List[int]]:
        if len(traj_indices) == 0:
            return {}
        st = self.dbscan_with_balltree(start_points)
        en = self.dbscan_with_balltree(end_points)
        return self.form_initial_groups(st, en, traj_indices)
