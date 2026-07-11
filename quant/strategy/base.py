"""
策略基类
定义策略生命周期和事件回调接口
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from quant.backtest.events import (
    Direction,
    FillEvent,
    MarketEvent,
    PortfolioEvent,
    SignalEvent,
)


class BaseStrategy(ABC):
    """策略基类

    生命周期:
        1. on_init()           — 策略初始化
        2. on_bar(market)      — 收到每根K线
        3. generate_signals()  — 生成交易信号
        4. on_fill(fill)       — 成交回报
        5. on_day_end(date, p) — 日终处理

    用法:
        class MyStrategy(BaseStrategy):
            name = "我的策略"
            params = {"fast": 5, "slow": 20}

            def on_init(self):
                self.ma_fast = []
                self.ma_slow = []

            def on_bar(self, market: MarketEvent):
                # 更新指标
                ...

            def generate_signals(self, date) -> List[SignalEvent]:
                # 生成信号
                ...
    """

    name: str = "base_strategy"
    version: str = "1.0"
    params: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        self.params = {**self.__class__.params, **kwargs}
        for k, v in self.params.items():
            setattr(self, k, v)

        self._engine = None
        self._date: Optional[date] = None
        self._positions: Dict[str, dict] = {}
        self._cash: float = 0
        self._history: Dict[str, List] = {}      # {symbol: [MarketEvent, ...]}
        self._signals: List[SignalEvent] = []

        logger.debug(f"🎯 策略初始化: {self.name} v{self.version}")

    # ─── 生命周期回调 ───────────────────────────────
    @abstractmethod
    def on_init(self) -> None:
        """策略初始化 — 设置指标、参数等"""
        ...

    @abstractmethod
    def on_bar(self, market: MarketEvent) -> None:
        """行情回调 — 每根K线触发

        Args:
            market: 单只股票的MarketEvent
        """
        ...

    @abstractmethod
    def generate_signals(self, current_date: date) -> List[SignalEvent]:
        """生成交易信号

        Args:
            current_date: 当前回测日期

        Returns:
            信号列表
        """
        ...

    def on_fill(self, fill: FillEvent) -> None:
        """成交回报回调 — 可选覆写"""
        pass

    def on_day_end(self, current_date: date, portfolio: PortfolioEvent) -> None:
        """日终处理回调 — 可选覆写"""
        pass

    # ─── 辅助方法 ───────────────────────────────────
    def buy(
        self,
        symbol: str,
        price: float,
        quantity: Optional[int] = None,
        ratio: float = 0.20,
        reason: str = "",
    ) -> SignalEvent:
        """生成买入信号"""
        return SignalEvent(
            symbol=symbol,
            date=self._date or date.today(),
            direction=Direction.BUY,
            strength=ratio,
            reason=reason or f"{self.name}买入",
            target_weight=ratio,
        )

    def sell(
        self,
        symbol: str,
        price: float,
        quantity: Optional[int] = None,
        ratio: float = 1.0,
        reason: str = "",
    ) -> SignalEvent:
        """生成卖出信号"""
        return SignalEvent(
            symbol=symbol,
            date=self._date or date.today(),
            direction=Direction.SELL,
            strength=ratio,
            reason=reason or f"{self.name}卖出",
        )

    def get_history(self, symbol: str, n: int = 20) -> List[MarketEvent]:
        """获取某只股票最近 N 条行情"""
        events = self._history.get(symbol, [])
        return events[-n:]

    def get_close_series(self, symbol: str, n: int = 20) -> List[float]:
        """获取最近 N 个收盘价"""
        return [e.close for e in self.get_history(symbol, n)]

    def get_position(self, symbol: str) -> dict:
        """获取某只股票的持仓信息"""
        if self._engine:
            return self._engine.broker.positions.get(symbol, {})
        return {}

    @property
    def cash(self) -> float:
        """可用资金"""
        if self._engine:
            return self._engine.broker.cash
        return self._cash

    @property
    def total_value(self) -> float:
        """总资产"""
        if self._engine:
            return self._engine.broker.total_value
        return self._cash
