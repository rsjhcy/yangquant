"""
投资组合管理
跟踪持仓、计算组合指标、支持权重再平衡
"""

from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant.backtest.events import FillEvent, MarketEvent, PortfolioEvent


class Portfolio:
    """投资组合管理器

    职责:
    - 跟踪持仓和现金
    - 计算组合市值/pnl
    - 记录净值曲线
    - 权重再平衡信号
    """

    def __init__(self, initial_cash: float = 1_000_000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, dict] = {}
        self.equity_curve: List[PortfolioEvent] = []

        # 交易记录
        self.trade_log: List[dict] = []

    def update_from_broker(
        self,
        cash: float,
        positions: Dict[str, dict],
    ) -> None:
        """从券商同步资金和持仓"""
        self.cash = cash
        self.positions = positions

    def mark_to_market(self, market_events: Dict[str, MarketEvent], current_date: date) -> PortfolioEvent:
        """按市价估值

        Args:
            market_events: {symbol: MarketEvent} 当日行情
            current_date: 当前日期

        Returns:
            PortfolioEvent 当日组合快照
        """
        market_value = 0.0
        positions_snapshot = {}

        for symbol, pos in list(self.positions.items()):
            qty = pos.get("quantity", 0)
            if qty <= 0:
                continue

            if symbol in market_events:
                price = market_events[symbol].close
            else:
                price = pos.get("avg_cost", 0)

            mv = qty * price
            market_value += mv
            positions_snapshot[symbol] = {
                "quantity": qty,
                "avg_cost": pos.get("avg_cost", 0),
                "market_price": price,
                "market_value": mv,
                "unrealized_pnl": mv - qty * pos.get("avg_cost", 0),
            }

        total_value = self.cash + market_value

        # 计算当日盈亏
        if self.equity_curve:
            prev_total = self.equity_curve[-1].total_value
            daily_pnl = total_value - prev_total
        else:
            daily_pnl = total_value - self.initial_cash

        cumulative_pnl = total_value - self.initial_cash
        cumulative_return = (total_value / self.initial_cash - 1) if self.initial_cash > 0 else 0

        event = PortfolioEvent(
            date=current_date,
            cash=self.cash,
            market_value=market_value,
            total_value=total_value,
            pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            cumulative_return=cumulative_return,
            positions=positions_snapshot,
        )

        self.equity_curve.append(event)
        return event

    def apply_fill(self, fill: FillEvent) -> None:
        """记录成交"""
        self.trade_log.append({
            "date": fill.date,
            "symbol": fill.symbol,
            "direction": fill.direction.value,
            "quantity": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
            "stamp_duty": fill.stamp_duty,
            "total_cost": fill.total_cost,
        })

    # ─── 组合指标 ───────────────────────────────────
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前持仓权重"""
        total = self.total_value if self.total_value > 0 else 1.0
        weights = {}
        for symbol, pos in self.positions.items():
            weights[symbol] = pos.get("market_value", 0) / total
        weights["_cash"] = self.cash / total
        return weights

    def calc_rebalance_orders(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        current_date: date,
    ) -> List[dict]:
        """计算再平衡所需订单

        Args:
            target_weights: {symbol: target_weight} 目标权重
            prices: {symbol: current_price} 当前价格
            current_date: 当前日期

        Returns:
            [{"symbol": ..., "direction": ..., "quantity": ..., "price": ...}, ...]
        """
        total_value = self.total_value
        orders = []

        all_symbols = set(target_weights.keys()) | set(self.positions.keys())

        for symbol in all_symbols:
            if symbol == "_cash":
                continue

            target_w = target_weights.get(symbol, 0)
            price = prices.get(symbol, 0)

            if price <= 0:
                continue

            current_qty = self.positions.get(symbol, {}).get("quantity", 0)
            target_qty = int(total_value * target_w / price / 100) * 100

            diff = target_qty - current_qty
            if diff == 0:
                continue

            from quant.backtest.events import Direction

            orders.append({
                "symbol": symbol,
                "direction": Direction.BUY if diff > 0 else Direction.SELL,
                "quantity": abs(diff),
                "price": price,
                "date": current_date,
            })

        return orders

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.get("market_value", 0) for p in self.positions.values())

    @property
    def equity_df(self) -> pd.DataFrame:
        """净值曲线 DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "date": e.date,
                "total_value": e.total_value,
                "cash": e.cash,
                "market_value": e.market_value,
                "pnl": e.pnl,
                "cumulative_return": e.cumulative_return,
            }
            for e in self.equity_curve
        ])
