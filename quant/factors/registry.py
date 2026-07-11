"""
因子注册中心
管理所有因子，提供统一的因子计算和缓存
"""

from typing import Dict, List, Optional, Type

import pandas as pd
from loguru import logger

from quant.factors.base import BaseFactor, FactorResult


class FactorRegistry:
    """因子注册中心 — 单例

    用法:
        registry = FactorRegistry()
        registry.register(my_factor)
        results = registry.compute_all(data, date)
    """

    _instance: Optional["FactorRegistry"] = None

    def __new__(cls) -> "FactorRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._factors: Dict[str, BaseFactor] = {}
            cls._instance._aliases: Dict[str, str] = {}
        return cls._instance

    def register(self, factor: BaseFactor) -> "FactorRegistry":
        """注册因子"""
        self._factors[factor.name] = factor
        logger.debug(f"📌 注册因子: {factor}")
        return self

    def register_many(self, factors: List[BaseFactor]) -> "FactorRegistry":
        """批量注册因子"""
        for f in factors:
            self.register(f)
        return self

    def unregister(self, name: str) -> bool:
        """注销因子"""
        if name in self._factors:
            del self._factors[name]
            return True
        return False

    def get(self, name: str) -> Optional[BaseFactor]:
        """获取因子"""
        return self._factors.get(name)

    def list_all(self) -> List[str]:
        """列出所有已注册因子名"""
        return list(self._factors.keys())

    def list_by_category(self, category=None) -> Dict[str, List[str]]:
        """按类别列出因子"""
        from quant.factors.base import FactorCategory
        grouped: Dict[str, List[str]] = {}
        for name, f in self._factors.items():
            cat = f.category.value
            if category and cat != category.value:
                continue
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(name)
        return grouped

    def compute_all(
        self,
        data: pd.DataFrame,
        date: Optional = None,
        names: Optional[List[str]] = None,
    ) -> Dict[str, FactorResult]:
        """计算一批因子

        Args:
            data: 行情面板数据
            date: 当前日期 (如不提供, 取data最后日期)
            names: 要计算的因子名 (None=全部)

        Returns:
            {factor_name: FactorResult}
        """
        target_names = names or list(self._factors.keys())
        results = {}

        for name in target_names:
            factor = self._factors.get(name)
            if factor is None:
                logger.warning(f"因子 '{name}' 未注册, 跳过")
                continue
            try:
                result = factor.compute(data)
                if date:
                    result.date = date
                results[name] = result
            except Exception as e:
                logger.error(f"计算因子 '{name}' 失败: {e}")

        return results

    def to_dataframe(self, results: Dict[str, FactorResult]) -> pd.DataFrame:
        """将因子计算结果合并为 DataFrame"""
        frames = []
        for name, result in results.items():
            s = result.values.copy()
            s.name = name
            frames.append(s)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)


# 全局实例
registry = FactorRegistry()
