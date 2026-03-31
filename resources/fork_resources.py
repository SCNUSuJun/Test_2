"""
多分叉河段依赖资源接口（方案步骤 11.4）。
实现可为文件/数据库/空间索引；此处仅定义读写契约，算法体可后续填充。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple


class ChannelGeometryStore(ABC):
    """航道中心线/缓冲区等几何数据，用于预测偏差到航道距离。"""

    @abstractmethod
    def distance_to_centerline_m(
        self, lon: float, lat: float, channel_id: str
    ) -> float:
        """返回点到指定航道中心线的垂直距离（米）；无法查询时返回 NaN 或抛错由实现约定。"""
        ...

    @abstractmethod
    def list_channel_ids_near(self, lon: float, lat: float, radius_m: float) -> List[str]:
        """研究区内与 (lon,lat) 邻近的航道标识列表。"""
        ...


class MmsiBranchStatsStore(ABC):
    """MMSI 历史轨迹在各聚类类别上的计数，用于分叉口分支概率。"""

    @abstractmethod
    def counts_for_mmsi_clusters(
        self, mmsi: int, cluster_ids: List[int]
    ) -> Dict[int, int]:
        """返回该 MMSI 在历史轨迹库中属于各 cluster_id 的条数 N_k。"""
        ...

    @abstractmethod
    def record_trajectory_assignment(
        self, mmsi: int, cluster_id: int, traj_id: str
    ) -> None:
        """离线或在线更新统计（持久化策略由实现决定）。"""
        ...


class ForkJunctionRegistry(ABC):
    """分叉口与可能分支类别集合的注册表。"""

    @abstractmethod
    def clusters_at_junction(self, junction_id: str) -> Set[int]:
        """给定分叉口 ID，返回可能涉及的聚类类别编号集合。"""
        ...

    @abstractmethod
    def junction_for_location(
        self, lon: float, lat: float
    ) -> Optional[str]:
        """若 (lon,lat) 落在已知分叉口缓冲区内则返回 junction_id，否则 None。"""
        ...
