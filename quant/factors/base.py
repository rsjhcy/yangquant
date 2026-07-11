"""
因子基类
统一因子接口 + 因子元数据
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class FactorCategory(Enum):
    """因子类别"""
    MOMENTUM = "动量"
    VOLATILITY = "波动"
    VOLUME_PRICE = "量价"
    TREND = "趋势"
    FUNDAMENTAL = "基本面"
    ALTERNATIVE = "另类"


class FactorDirection(Enum):
    """因子方向"""
    POSITIVE = 1    # 正向(值越大越好)
    NEGATIVE = -1   # 负向


@dataclass
class FactorResult:
    """因子计算结果"""
    name: str
    values: pd.Series                   # index=symbol, values=factor_value
    date: date
    category: FactorCategory
    direction: FactorDirection
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFactor(ABC):
    """因子基类 — 所有因子必须继承

    用法:
        class MyFactor(BaseFactor):
            name = "my_factor"
            category = FactorCategory.MOMENTUM
            direction = FactorDirection.POSITIVE
            requires = ["close", "volume"]

            def compute(self, data: dict) -> FactorResult:
                ...
    """

    name: str = "base_factor"
    category: FactorCategory = FactorCategory.MOMENTUM
    direction: FactorDirection = FactorDirection.POSITIVE
    requires: List[str] = ["close"]              # 需要的字段
    lookback: int = 20                           # 默认回溯期
    description: str = ""

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> FactorResult:
        """计算因子值

        Args:
            data: DataFrame with columns: symbol, date, open, close, high, low, volume, ...
                  必须是单日截面或多日面板数据

        Returns:
            FactorResult with values indexed by symbol
        """
        ...

    def validate_input(self, data: pd.DataFrame) -> None:
        """校验输入数据包含所需字段"""
        missing = [c for c in self.requires if c not in data.columns]
        if missing:
            raise ValueError(f"因子 [{self.name}] 需要字段 {missing}，数据只有: {list(data.columns)}")

    def __repr__(self) -> str:
        return f"Factor({self.name}, {self.category.value}, {self.direction.name})"


# ─── 内置因子工厂函数 ─────────────────────────────

def make_rolling_factor(
    name: str,
    field: str,
    func_name: str,
    lookback: int = 20,
    category: FactorCategory = FactorCategory.MOMENTUM,
    direction: FactorDirection = FactorDirection.POSITIVE,
    **func_kwargs,
) -> BaseFactor:
    """快速创建滚动窗口因子

    Args:
        field: 数据字段名 (如 'close')
        func_name: rolling后的聚合函数名 ('mean', 'std', 'max', 'min'...)
        lookback: 窗口长度
        category: 因子类别
        direction: 因子方向
    """

    class _RollingFactor(BaseFactor):
        """动态生成滚动因子"""
        name = name
        category = category
        direction = direction
        requires = [field]
        lookback = lookback

        @property
        def description(self) -> str:
            return f"{field}的{lookback}日{func_name}"

        def compute(self, data: pd.DataFrame) -> FactorResult:
            self.validate_input(data)
            values = data.groupby("symbol")[field].transform(
                lambda x: getattr(x.rolling(lookback, min_periods=lookback // 2), func_name)(**func_kwargs)
            )
            return FactorResult(
                name=self.name,
                values=values,
                date=data["date"].iloc[-1] if "date" in data.columns else date.today(),
                category=self.category,
                direction=self.direction,
            )

    return _RollingFactor()
