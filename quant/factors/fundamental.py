"""
基本面因子
估值因子 / 盈利因子 / 成长因子 / 质量因子

注意: 基本面数据来自 akshare 财务接口，更新频率为季度。
对于日频回测，因子值在财报发布日更新，其余时间用最近值填充。
"""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from quant.factors.base import (
    BaseFactor,
    FactorCategory,
    FactorDirection,
    FactorResult,
)


class ValuationFactor(BaseFactor):
    """估值因子 — PE/PB/PS 分位

    基于历史分位数，低估值=正向信号
    """

    name = "valuation_percentile"
    category = FactorCategory.FUNDAMENTAL
    direction = FactorDirection.POSITIVE  # 低估值好 = 分位值越低越好

    def __init__(self, metric: str = "pe"):
        """
        Args:
            metric: 'pe' | 'pb' | 'ps'
        """
        self.metric = metric
        self.name = f"{metric}_percentile"

    def compute(self, data: pd.DataFrame) -> FactorResult:
        """需要包含 PE/PB/PS 历史数据的 DataFrame"""
        required_col = self.metric.upper() if self.metric in ("pe", "pb", "ps") else self.metric

        results = {}
        for symbol, group in data.groupby("symbol"):
            if required_col not in group.columns:
                results[symbol] = np.nan
                continue

            values = group[required_col].dropna()
            if len(values) < 20:
                results[symbol] = np.nan
                continue

            # 当前值在历史上的分位
            current = values.iloc[-1]
            percentile = (values < current).mean()
            results[symbol] = percentile  # 低分位=低估值=好

        return FactorResult(
            name=self.name,
            values=pd.Series(results),
            date=data["date"].iloc[-1] if "date" in data.columns else date.today(),
            category=self.category,
            direction=self.direction,
        )


class ProfitabilityFactor(BaseFactor):
    """盈利能力因子 — ROE"""

    name = "roe"
    category = FactorCategory.FUNDAMENTAL
    direction = FactorDirection.POSITIVE  # ROE越高越好

    def compute(self, data: pd.DataFrame) -> FactorResult:
        results = {}
        roe_col = None
        for c in ["ROE", "roe", "净资产收益率"]:
            if c in data.columns:
                roe_col = c
                break

        if roe_col is None:
            return FactorResult(
                name=self.name,
                values=pd.Series(dtype=float),
                date=date.today(),
                category=self.category,
                direction=self.direction,
            )

        for symbol, group in data.groupby("symbol"):
            values = group[roe_col].dropna()
            if len(values) > 0:
                results[symbol] = values.iloc[-1]
            else:
                results[symbol] = np.nan

        return FactorResult(
            name=self.name,
            values=pd.Series(results),
            date=data["date"].iloc[-1] if "date" in data.columns else date.today(),
            category=self.category,
            direction=self.direction,
        )


class GrowthFactor(BaseFactor):
    """成长因子 — 营收增速"""

    name = "revenue_growth"
    category = FactorCategory.FUNDAMENTAL
    direction = FactorDirection.POSITIVE  # 增速越高越好

    def compute(self, data: pd.DataFrame) -> FactorResult:
        results = {}
        rev_col = None
        for c in ["revenue_yoy", "营业收入同比增长率"]:
            if c in data.columns:
                rev_col = c
                break

        if rev_col is None:
            return FactorResult(
                name=self.name,
                values=pd.Series(dtype=float),
                date=date.today(),
                category=self.category,
                direction=self.direction,
            )

        for symbol, group in data.groupby("symbol"):
            values = group[rev_col].dropna()
            if len(values) > 0:
                results[symbol] = values.iloc[-1]
            else:
                results[symbol] = np.nan

        return FactorResult(
            name=self.name,
            values=pd.Series(results),
            date=data["date"].iloc[-1] if "date" in data.columns else date.today(),
            category=self.category,
            direction=self.direction,
        )
