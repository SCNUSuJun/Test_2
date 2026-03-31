"""
步骤6A：提取每条轨迹的起点和终点坐标
======================================
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from utils import setup_logger


class EndpointExtractor:
    """轨迹起终点提取器"""

    def __init__(self):
        self.logger = setup_logger("EndpointExtractor")

    def extract(
        self, trajectories: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        starts = []
        ends = []
        indices = []
        for idx, t in enumerate(trajectories):
            if t is None or len(t) < 1:
                continue
            g = t.sort_values("Timestamp", kind="mergesort")
            starts.append([float(g.iloc[0]["LON"]), float(g.iloc[0]["LAT"])])
            ends.append([float(g.iloc[-1]["LON"]), float(g.iloc[-1]["LAT"])])
            indices.append(idx)
        if not indices:
            return (
                np.zeros((0, 2)),
                np.zeros((0, 2)),
                [],
            )
        return np.array(starts), np.array(ends), indices
