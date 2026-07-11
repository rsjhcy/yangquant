"""
双均线交叉策略
经典趋势跟踪策略: 快线上穿慢线 → 买入; 快线下穿慢线 → 卖出
"""

from collections import defaultdict
from datetime import date
from typing import Dict, List

from loguru import logger

from quant.backtest.events import (
    Direction,
    MarketEvent,
    SignalEvent,
)
from quant.strategy.base import BaseStrategy
from quant.strategy.signals import SignalGenerator


class MACrossoverStrategy(BaseStrategy):
    """双均线交叉策略

    参数:
        fast: 快线周期 (默认5日)
        slow: 慢线周期 (默认20日)
        use_volume_filter: 是否用成交量过滤假突破

    逻辑:
        - 金叉(快线上穿慢线) → 全仓买入
        - 死叉(快线下穿慢线) → 全部卖出
        - 每只股票独立运行
    """

    name = "双均线交叉"
    version = "1.0"
    params = {
        "fast": 5,
        "slow": 20,
        "use_volume_filter": False,
    }

    def on_init(self) -> None:
        self._closes: Dict[str, List[float]] = defaultdict(list)
        self._volumes: Dict[str, List[float]] = defaultdict(list)
        self._positions: set = set()  # 已持仓股票

    def on_bar(self, market: MarketEvent) -> None:
        symbol = market.symbol
        self._closes[symbol].append(market.close)
        self._volumes[symbol].append(market.volume)

        # 保持列表长度 (只保留需要的历史数据)
        max_len = self.slow + 10
        if len(self._closes[symbol]) > max_len:
            self._closes[symbol] = self._closes[symbol][-max_len:]
            self._volumes[symbol] = self._volumes[symbol][-max_len:]

    def generate_signals(self, current_date: date) -> List[SignalEvent]:
        signals = []

        for symbol, closes in self._closes.items():
            if len(closes) < self.slow + 1:
                continue

            # 计算均线
            fast_ma = SignalGenerator.sma(closes, self.fast)
            slow_ma = SignalGenerator.sma(closes, self.slow)

            # 检测交叉
            cross_type, idx = SignalGenerator.detect_cross(
                fast_ma.tolist(), slow_ma.tolist()
            )

            if cross_type is None:
                continue

            # 最新的收盘价
            price = closes[-1]

            # 成交量过滤 (可选)
            if self.use_volume_filter:
                volumes = self._volumes.get(symbol, [])
                if len(volumes) >= 5:
                    avg_vol = sum(volumes[-6:-1]) / 5
                    if volumes[-1] < avg_vol * 1.2:
                        continue  # 成交量不够大，视为假突破

            if cross_type == "golden_cross":
                if symbol not in self._positions:
                    signals.append(
                        self.buy(
                            symbol=symbol,
                            price=price,
                            ratio=0.25,  # 单只股票最多25%仓位
                            reason=f"金叉: MA{self.fast}↑MA{self.slow}",
                        )
                    )
                    self._positions.add(symbol)

            elif cross_type == "death_cross":
                if symbol in self._positions:
                    signals.append(
                        self.sell(
                            symbol=symbol,
                            price=price,
                            ratio=1.0,  # 全部卖出
                            reason=f"死叉: MA{self.fast}↓MA{self.slow}",
                        )
                    )
                    self._positions.discard(symbol)

        return signals
