"""
回测引擎主循环
事件驱动架构，按交易日推进行情、策略信号、订单匹配、组合估值
"""

from datetime import date, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant.backtest.broker import SimulatedBroker
from quant.backtest.events import (
    Direction,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    PortfolioEvent,
    SettlementEvent,
    SignalEvent,
)
from quant.backtest.portfolio import Portfolio
from quant.data.calendar import cal


class BacktestEngine:
    """事件驱动回测引擎

    用法:
        engine = BacktestEngine(
            initial_cash=1_000_000,
            commission_rate=0.00025,
        )

        engine.set_data(daily_data)  # DataFrame with OHLCV
        engine.set_strategy(my_strategy)  # 策略对象

        result = engine.run()
        engine.report()
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000,
        commission_rate: float = 0.00025,
        stamp_duty: float = 0.001,
        slippage: float = 0.0001,
        benchmark: Optional[str] = None,
    ):
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.slippage = slippage
        self.benchmark = benchmark

        # 核心组件
        self.broker = SimulatedBroker(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            stamp_duty=stamp_duty,
            slippage=slippage,
        )
        self.portfolio = Portfolio(initial_cash=initial_cash)

        # 数据
        self._data: pd.DataFrame = pd.DataFrame()
        self._data_cache: Dict[date, Dict[str, MarketEvent]] = {}
        self._dates: List[date] = []
        self._symbols: List[str] = []

        # 策略
        self._strategy = None
        self._signal_handler: Optional[Callable] = None
        self._risk_handler: Optional[Callable] = None

        # 回调
        self._on_bar_callbacks: List[Callable] = []
        self._on_fill_callbacks: List[Callable] = []
        self._on_day_end_callbacks: List[Callable] = []

        # 结果
        self.result: Optional[BacktestResult] = None
        self._running = False

    # ─── 数据准备 ───────────────────────────────────
    def set_data(self, df: pd.DataFrame) -> None:
        """设置行情数据

        Args:
            df: DataFrame with columns: symbol, date, open, high, low, close, volume, ...
        """
        required_cols = ["symbol", "date", "open", "high", "low", "close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"数据缺少必要列: {missing}")

        self._data = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self._data["date"]):
            self._data["date"] = pd.to_datetime(self._data["date"]).dt.date

        self._data = self._data.sort_values("date").reset_index(drop=True)
        self._symbols = sorted(self._data["symbol"].unique().tolist())
        self._dates = sorted(self._data["date"].unique().tolist())

        # 构建行情缓存 {date: {symbol: MarketEvent}}
        self._build_market_cache()

        logger.info(
            f"📊 回测数据加载: {len(self._symbols)} 只股票, "
            f"{len(self._dates)} 个交易日 "
            f"({self._dates[0]} ~ {self._dates[-1]})"
        )

    def _build_market_cache(self) -> None:
        """预构建行情事件缓存"""
        self._data_cache.clear()

        # 构建前收盘价映射
        prev_closes: Dict[str, float] = {}

        for dt in self._dates:
            day_data = self._data[self._data["date"] == dt]
            day_events: Dict[str, MarketEvent] = {}

            for _, row in day_data.iterrows():
                symbol = str(row["symbol"])
                close = float(row["close"])
                pre_close = prev_closes.get(symbol, close)
                volume = float(row.get("volume", 0)) if pd.notna(row.get("volume", float('nan'))) else 0.0

                # 判断涨停/跌停
                rate = 0.10
                if symbol.startswith("3"):  # 创业板20%
                    rate = 0.20
                limit_up_price = round(pre_close * (1 + rate), 2)
                limit_down_price = round(pre_close * (1 - rate), 2)

                event = MarketEvent(
                    symbol=symbol,
                    date=dt,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=close,
                    volume=volume,
                    amount=float(row.get("amount", 0)) if pd.notna(row.get("amount", float('nan'))) else 0.0,
                    turnover=float(row.get("turnover", 0)) if pd.notna(row.get("turnover", float('nan'))) else 0.0,
                    pre_close=pre_close,
                    is_limit_up=close >= limit_up_price - 0.005,
                    is_limit_down=close <= limit_down_price + 0.005,
                )

                day_events[symbol] = event
                prev_closes[symbol] = close

            self._data_cache[dt] = day_events

    # ─── 策略 & 信号 ────────────────────────────────
    def set_strategy(self, strategy) -> None:
        """设置交易策略"""
        self._strategy = strategy
        strategy._engine = self
        logger.info(f"🎯 策略已设置: {strategy.name}")

    def on_bar(self, callback: Callable) -> Callable:
        """注册行情回调"""
        self._on_bar_callbacks.append(callback)
        return callback

    def on_fill(self, callback: Callable) -> Callable:
        """注册成交回调"""
        self._on_fill_callbacks.append(callback)
        return callback

    # ─── 主循环 ────────────────────────────────────
    def run(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        progress: bool = True,
    ) -> "BacktestResult":
        """运行回测

        Args:
            start_date: 回测起始日期
            end_date: 回测结束日期
            progress: 是否显示进度

        Returns:
            BacktestResult 回测结果
        """
        if self._strategy is None:
            raise ValueError("请先设置策略: engine.set_strategy(strategy)")

        if not self._data_cache:
            raise ValueError("请先设置行情数据: engine.set_data(df)")

        # 过滤日期范围
        dates = self._dates
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        logger.info(f"🚀 开始回测: {len(dates)} 个交易日 ({dates[0]} ~ {dates[-1]})")
        self._running = True

        # 初始化策略
        self._strategy.on_init()

        for i, dt in enumerate(dates):
            if not self._running:
                break

            day_events = self._data_cache[dt]

            if not day_events:
                continue

            # 1. 行情驱动
            for symbol, market_event in day_events.items():
                self._strategy.on_bar(market_event)
                for cb in self._on_bar_callbacks:
                    cb(market_event)

            # 2. 策略生成信号
            signals = self._strategy.generate_signals(dt)

            # 3. 信号 → 订单
            for signal in signals:
                order = self.broker.create_order(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    quantity=self._calc_order_quantity(signal, day_events),
                    price=day_events.get(signal.symbol, MarketEvent()).close,
                    current_date=dt,
                    reason=signal.reason,
                )
                if order.status.name != "REJECTED":
                    self.broker.submit_order(order)

            # 4. 订单匹配 → 成交
            fills = self.broker.match_orders(day_events)

            # 5. 成交处理
            for fill in fills:
                self.portfolio.apply_fill(fill)
                self._strategy.on_fill(fill)
                for cb in self._on_fill_callbacks:
                    cb(fill)

            # 6. 日终估值
            self.portfolio.update_from_broker(self.broker.cash, self.broker.positions)
            portfolio_event = self.portfolio.mark_to_market(day_events, dt)

            # 7. 日终处理
            self.broker.end_of_day(dt)
            self._strategy.on_day_end(dt, portfolio_event)
            for cb in self._on_day_end_callbacks:
                cb(dt, portfolio_event)

            # 进度
            if progress and (i + 1) % 50 == 0:
                nav = portfolio_event.total_value
                ret = portfolio_event.cumulative_return
                logger.info(f"  📈 [{i+1}/{len(dates)}] {dt} | 净值: {nav:,.0f} | 收益: {ret:+.2%}")

        # 生成结果
        self.result = BacktestResult(
            equity_curve=self.portfolio.equity_df,
            trade_log=pd.DataFrame(self.portfolio.trade_log),
            initial_cash=self.initial_cash,
            symbols=self._symbols,
            benchmark=self.benchmark,
        )

        logger.info(f"✅ 回测完成! 最终净值: {self.portfolio.total_value:,.0f}")
        return self.result

    def _calc_order_quantity(
        self, signal: SignalEvent, day_events: Dict[str, MarketEvent]
    ) -> int:
        """根据信号计算订单数量"""
        from quant.backtest.constraints import round_lot

        price = day_events.get(signal.symbol, MarketEvent()).close
        if price <= 0:
            return 0

        if signal.target_weight > 0:
            # 基于目标权重计算
            target_value = self.portfolio.total_value * signal.target_weight
            return round_lot(int(target_value / price))
        elif signal.strength > 0:
            # 基于信号强度按比例买
            max_qty = round_lot(int(self.broker.cash * signal.strength / price / 10))
            return min(max_qty, round_lot(int(self.broker.cash * 0.25 / price)))
        else:
            # 默认: 买入可用资金的20%
            return round_lot(int(self.broker.cash * 0.20 / price))

    # ─── 停止 & 快照 ────────────────────────────────
    def stop(self) -> None:
        """停止回测"""
        self._running = False

    def snapshot(self) -> dict:
        """当前状态快照"""
        return {
            "cash": self.broker.cash,
            "market_value": self.broker.market_value,
            "total_value": self.broker.total_value,
            "positions": self.broker.positions,
            "unrealized_pnl": self.broker.market_value - sum(
                p["quantity"] * p["avg_cost"] for p in self.broker.positions.values()
            ),
        }


class BacktestResult:
    """回测结果"""

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
        initial_cash: float,
        symbols: List[str],
        benchmark: Optional[str] = None,
    ):
        self.equity_curve = equity_curve
        self.trade_log = trade_log
        self.initial_cash = initial_cash
        self.symbols = symbols
        self.benchmark = benchmark

    @property
    def final_value(self) -> float:
        if self.equity_curve.empty:
            return self.initial_cash
        return self.equity_curve["total_value"].iloc[-1]

    @property
    def total_return(self) -> float:
        return (self.final_value / self.initial_cash) - 1

    @property
    def trade_count(self) -> int:
        return len(self.trade_log)

    @property
    def win_rate(self) -> float:
        """粗略胜率（基于交易盈亏）"""
        if self.trade_log.empty:
            return 0.0
        sells = self.trade_log[self.trade_log["direction"] == "SELL"]
        if sells.empty:
            return 0.5  # 无平仓，假设持平
        # 简化：用卖出量×价格 vs 买入量×成本
        return 0.5  # 需要配对交易才能精确计算

    def describe(self) -> dict:
        """描述性统计"""
        if self.equity_curve.empty:
            return {"error": "无回测数据"}

        returns = self.equity_curve["pnl"] / self.equity_curve["total_value"].shift(1)
        returns = returns.dropna()

        return {
            "initial_capital": self.initial_cash,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "trade_count": self.trade_count,
            "max_drawdown": self._calc_max_drawdown(),
            "annual_return": self._calc_annual_return(),
            "sharpe_ratio": self._calc_sharpe(returns),
            "volatility": float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0,
            "calmar_ratio": self._calc_calmar(),
        }

    def _calc_max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        values = self.equity_curve["total_value"].values
        peak = np.maximum.accumulate(values)
        dd = (values - peak) / peak
        return float(dd.min())

    def _calc_annual_return(self) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        total = self.total_return
        days = (self.equity_curve["date"].iloc[-1] - self.equity_curve["date"].iloc[0]).days
        if days <= 0:
            return 0.0
        years = days / 365.25
        return float((1 + total) ** (1 / max(years, 0.01)) - 1)

    def _calc_sharpe(self, returns: pd.Series) -> float:
        if len(returns) < 2:
            return 0.0
        rf_daily = 0.02 / 252  # 假设无风险利率2%
        excess = returns - rf_daily
        mean = excess.mean()
        std = excess.std()
        if std == 0:
            return 0.0
        return float(mean / std * np.sqrt(252))

    def _calc_calmar(self) -> float:
        ann_ret = self._calc_annual_return()
        mdd = self._calc_max_drawdown()
        if abs(mdd) < 1e-6:
            return 0.0
        return ann_ret / abs(mdd)
