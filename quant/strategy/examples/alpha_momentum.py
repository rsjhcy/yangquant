"""
多因子动量轮动策略
计算多因子综合得分，买入得分最高的N只股票，定期调仓
"""

from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant.backtest.events import (
    Direction,
    MarketEvent,
    PortfolioEvent,
    SignalEvent,
)
from quant.strategy.base import BaseStrategy


class AlphaMomentumStrategy(BaseStrategy):
    """多因子动量轮动策略

    参数:
        top_n: 持仓数量 (默认10只)
        rebalance_days: 调仓周期 (默认5个交易日)
        momentum_period: 动量计算周期 (默认20日)
        volume_period: 成交量平均周期 (默认20日)
        min_volume_ratio: 最小成交量比率 (默认0.5, 排除流动性差的)

    逻辑:
        1. 每只股票计算综合得分:
           - 动量得分 (N日收益率)
           - 波动得分 (低波动加分)
           - 量价得分 (放量上涨加分)
        2. 按得分排序，买入 top_n 只
        3. 每隔 rebalance_days 天调仓
    """

    name = "多因子动量轮动"
    version = "1.0"
    params = {
        "top_n": 10,
        "rebalance_days": 5,
        "momentum_period": 20,
        "volume_period": 20,
        "min_volume_ratio": 0.5,
    }

    def on_init(self) -> None:
        self._closes: Dict[str, List[float]] = defaultdict(list)
        self._volumes: Dict[str, List[float]] = defaultdict(list)
        self._highs: Dict[str, List[float]] = defaultdict(list)
        self._lows: Dict[str, List[float]] = defaultdict(list)
        self._current_holdings: set = set()
        self._days_since_rebalance: int = 0

    def on_bar(self, market: MarketEvent) -> None:
        symbol = market.symbol
        self._closes[symbol].append(market.close)
        self._volumes[symbol].append(market.volume)
        self._highs[symbol].append(market.high)
        self._lows[symbol].append(market.low)

        max_len = max(self.momentum_period, self.volume_period) + 20
        for d in [self._closes, self._volumes, self._highs, self._lows]:
            if len(d[symbol]) > max_len:
                d[symbol] = d[symbol][-max_len:]

    def generate_signals(self, current_date: date) -> List[SignalEvent]:
        signals = []

        # 筛选有足够数据的股票
        eligible = {
            sym for sym, closes in self._closes.items()
            if len(closes) >= self.momentum_period and len(self._volumes.get(sym, [])) >= self.volume_period
        }

        if len(eligible) < self.top_n:
            return signals

        # 计算综合得分
        scores = {}
        for symbol in eligible:
            score = self._calc_score(symbol)
            if not np.isnan(score):
                scores[symbol] = score

        if len(scores) < self.top_n:
            return signals

        # 每年大约 20 个调仓日
        if self._days_since_rebalance < self.rebalance_days:
            self._days_since_rebalance += 1
            return signals

        self._days_since_rebalance = 0

        # 排序选股
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = set(sym for sym, _ in ranked[:self.top_n])

        # 卖出不再选中的
        for symbol in self._current_holdings - selected:
            pos = self.get_position(symbol)
            if pos.get("quantity", 0) > 0:
                signals.append(
                    self.sell(
                        symbol=symbol,
                        price=self._closes[symbol][-1],
                        ratio=1.0,
                        reason=f"调仓卖出: 得分掉落",
                    )
                )

        # 买入新选中的
        weight = 1.0 / self.top_n * 0.9  # 留10%现金
        for symbol in selected - self._current_holdings:
            signals.append(
                self.buy(
                    symbol=symbol,
                    price=self._closes[symbol][-1],
                    ratio=weight,
                    reason=f"调仓买入: 得分{scores[symbol]:.4f}",
                )
            )

        self._current_holdings = selected
        return signals

    def _calc_score(self, symbol: str) -> float:
        """计算综合因子得分"""
        closes = self._closes[symbol]
        volumes = self._volumes[symbol]
        highs = self._highs[symbol]
        lows = self._lows[symbol]

        period = self.momentum_period

        # 1. 动量得分 (N日收益率)
        mom = (closes[-1] / closes[-period] - 1) if closes[-period] > 0 else 0

        # 2. 波动得分 (低波动更好)
        recent_returns = np.diff(closes[-period:]) / np.array(closes[-period:-1])
        vol = np.std(recent_returns) if len(recent_returns) > 1 else 1.0
        vol_score = -vol  # 负值: 波动越低越好

        # 3. 量价得分 (近期成交量相对历史)
        recent_vol = np.mean(volumes[-5:])
        hist_vol = np.mean(volumes[-self.volume_period:])
        vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1.0

        # 4. 最大回撤惩罚
        drawdown = self._calc_drawdown(closes[-period:])
        dd_penalty = -abs(drawdown) * 2

        # 综合得分 (Z-score归一化后加权)
        # 权重: 动量40% + 低波动25% + 量价15% + 低回撤20%
        score = (
            mom * 0.40
            + vol_score * 0.25
            + (vol_ratio - 1) * 0.15
            + dd_penalty * 0.20
        )

        return score

    @staticmethod
    def _calc_drawdown(prices: List[float]) -> float:
        """计算序列的最大回撤"""
        values = np.array(prices)
        peak = np.maximum.accumulate(values)
        return float(np.min((values - peak) / peak))
