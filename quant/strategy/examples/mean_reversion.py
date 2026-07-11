"""
布林带均值回归策略
价格触及下轨 → 买入; 回归中轨 → 卖出; 触及上轨 → 做空(暂不支持)
"""

from collections import defaultdict
from datetime import date
from typing import Dict, List

import numpy as np

from quant.backtest.events import (
    Direction,
    MarketEvent,
    SignalEvent,
)
from quant.strategy.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """布林带均值回归策略

    参数:
        period: 布林带周期 (默认20)
        std_mult: 标准差倍数 (默认2.0)
        stop_loss: 止损比例 (默认0.05 = -5%)
        take_profit: 止盈比例 (默认0.10 = +10%)

    逻辑:
        - 价格跌破下轨 → 买入 (超卖)
        - 价格突破中轨 → 卖出 (回归)
        - 跌破买入价的95% → 止损
        - 达到买入价的110% → 止盈
    """

    name = "布林带均值回归"
    version = "1.0"
    params = {
        "period": 20,
        "std_mult": 2.0,
        "stop_loss": 0.05,
        "take_profit": 0.10,
    }

    def on_init(self) -> None:
        self._closes: Dict[str, List[float]] = defaultdict(list)
        # 记录每只股票的买入价和买入日期
        self._buy_info: Dict[str, dict] = {}

    def on_bar(self, market: MarketEvent) -> None:
        symbol = market.symbol
        self._closes[symbol].append(market.close)
        max_len = self.period + 20
        if len(self._closes[symbol]) > max_len:
            self._closes[symbol] = self._closes[symbol][-max_len:]

    def generate_signals(self, current_date: date) -> List[SignalEvent]:
        signals = []

        for symbol, closes in self._closes.items():
            if len(closes) < self.period:
                continue

            recent = closes[-self.period:]
            ma = np.mean(recent)
            std = np.std(recent)

            if std <= 0:
                continue

            upper = ma + self.std_mult * std
            lower = ma - self.std_mult * std
            current_price = closes[-1]

            # 判断是否持仓
            pos = self.get_position(symbol)
            has_position = pos.get("quantity", 0) > 0

            if has_position and symbol in self._buy_info:
                buy_price = self._buy_info[symbol]["price"]
                pnl_pct = (current_price - buy_price) / buy_price

                # 止损
                if pnl_pct <= -self.stop_loss:
                    signals.append(
                        self.sell(
                            symbol=symbol,
                            price=current_price,
                            ratio=1.0,
                            reason=f"止损: {pnl_pct:.1%}",
                        )
                    )
                    del self._buy_info[symbol]
                    continue

                # 止盈
                if pnl_pct >= self.take_profit:
                    signals.append(
                        self.sell(
                            symbol=symbol,
                            price=current_price,
                            ratio=1.0,
                            reason=f"止盈: {pnl_pct:.1%}",
                        )
                    )
                    del self._buy_info[symbol]
                    continue

                # 回归中轨
                if current_price >= ma:
                    signals.append(
                        self.sell(
                            symbol=symbol,
                            price=current_price,
                            ratio=1.0,
                            reason=f"回归MA: {current_price:.2f}≥{ma:.2f}",
                        )
                    )
                    del self._buy_info[symbol]
                    continue

            elif not has_position:
                # 触及下轨 → 买入
                if current_price <= lower:
                    signals.append(
                        self.buy(
                            symbol=symbol,
                            price=current_price,
                            ratio=0.20,
                            reason=f"触及下轨: {current_price:.2f}≤{lower:.2f}",
                        )
                    )
                    self._buy_info[symbol] = {
                        "price": current_price,
                        "date": current_date,
                    }

        return signals
