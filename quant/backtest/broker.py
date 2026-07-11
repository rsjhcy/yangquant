"""
模拟券商
处理订单匹配、滑点模拟、交易成本计算
A股专用: 涨跌停限制、T+1
"""

from datetime import date, datetime
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd
from loguru import logger

from quant.backtest.constraints import (
    AShareConstraints,
    calc_limit_price,
    is_limit_hit,
    round_lot,
)
from quant.backtest.events import (
    Direction,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderStatus,
    OrderType,
    SignalEvent,
)


class SimulatedBroker:
    """模拟券商 — 处理订单匹配和交易成本

    模拟真实的A股交易约束:
    - T+1: 当日买入次日可卖
    - 涨跌停: 涨停无法买入, 跌停无法卖出
    - 最小交易单位: 100股(1手)
    - 佣金: 万2.5双边, 最低5元
    - 印花税: 卖出0.1%
    - 滑点: 可配置
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000,
        commission_rate: float = 0.00025,
        stamp_duty: float = 0.001,
        min_commission: float = 5.0,
        slippage: float = 0.0001,  # 万1滑点
    ):
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.min_commission = min_commission
        self.slippage = slippage

        self.cash = initial_cash
        self.positions: Dict[str, dict] = {}      # {symbol: {qty, avg_cost, market_value, buy_date, sellable}}
        self.orders: Dict[str, OrderEvent] = {}    # 活跃订单
        self.order_history: List[OrderEvent] = []
        self.fill_history: List[FillEvent] = []

        # T+1 追踪: 每只股票今日买入数量
        self._today_buy: Dict[str, int] = {}
        # 买入日期记录 (用于T+1可用性判断)
        self._buy_records: Dict[str, List[tuple]] = {}  # {symbol: [(date, qty), ...]}

    # ─── 订单生成 ───────────────────────────────────
    def create_order(
        self,
        symbol: str,
        direction: Direction,
        quantity: int,
        price: float,
        order_type: OrderType = OrderType.LIMIT,
        current_date: Optional[date] = None,
        reason: str = "",
    ) -> OrderEvent:
        """创建订单"""
        order_id = f"ORD-{uuid4().hex[:8].upper()}"

        order = OrderEvent(
            symbol=symbol,
            date=current_date or date.today(),
            direction=direction,
            order_type=order_type,
            quantity=round_lot(quantity),
            price=price,
            order_id=order_id,
            status=OrderStatus.CREATED,
            reason=reason,
        )

        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            logger.warning(f"订单被拒绝(数量为0): {symbol} {direction.value}")

        return order

    def submit_order(self, order: OrderEvent) -> OrderEvent:
        """提交订单"""
        order.status = OrderStatus.SUBMITTED
        self.orders[order.order_id] = order
        self.order_history.append(order)
        return order

    # ─── 订单匹配核心 ───────────────────────────────
    def match_orders(self, market_events: Dict[str, MarketEvent]) -> List[FillEvent]:
        """用当日行情匹配订单，生成成交列表

        Args:
            market_events: {symbol: MarketEvent} 当日行情

        Returns:
            成交事件列表
        """
        fills = []
        rejected_orders = []

        for order_id, order in list(self.orders.items()):
            if order.symbol not in market_events:
                continue

            market = market_events[order.symbol]
            fill = self._match_single(order, market)

            if fill:
                fills.append(fill)
                self._apply_fill(fill)
                self.fill_history.append(fill)
                order.status = OrderStatus.FILLED
                del self.orders[order_id]
            elif order.status == OrderStatus.REJECTED:
                rejected_orders.append(order_id)
                del self.orders[order_id]

        if fills:
            logger.debug(f"📊 成交 {len(fills)} 笔")

        return fills

    def _match_single(self, order: OrderEvent, market: MarketEvent) -> Optional[FillEvent]:
        """匹配单个订单"""
        symbol = order.symbol

        # 停牌检查
        if market.is_suspended:
            logger.debug(f"{symbol} 停牌, 无法成交")
            order.status = OrderStatus.REJECTED
            return None

        # 涨跌停检查
        if order.direction == Direction.BUY and market.is_limit_up:
            logger.debug(f"{symbol} 涨停, 无法买入")
            order.status = OrderStatus.REJECTED
            return None

        if order.direction == Direction.SELL and market.is_limit_down:
            logger.debug(f"{symbol} 跌停, 无法卖出")
            order.status = OrderStatus.REJECTED
            return None

        # 确定成交价
        fill_price = self._determine_fill_price(order, market)

        # 涨价停限制检查
        pre_close = getattr(market, 'pre_close', market.close)
        is_st = symbol.startswith("300") is False and "ST" not in symbol  # 简化ST判断

        if order.direction == Direction.BUY:
            limit_up_price = calc_limit_price(pre_close, "up", is_st)
            if fill_price > limit_up_price:
                fill_price = limit_up_price
                logger.debug(f"{symbol} 买入价触及涨停价, 调整至 {limit_up_price}")

        if order.direction == Direction.SELL:
            limit_down_price = calc_limit_price(pre_close, "down", is_st)
            if fill_price < limit_down_price:
                fill_price = limit_down_price
                logger.debug(f"{symbol} 卖出价触及跌停价, 调整至 {limit_down_price}")

        # 资金检查(买入)
        if order.direction == Direction.BUY:
            estimated_cost = fill_price * order.quantity * (1 + self.commission_rate)
            if estimated_cost > self.cash:
                # 调整数量
                max_qty = int(self.cash / (fill_price * (1 + self.commission_rate)))
                order.quantity = round_lot(max_qty)
                if order.quantity <= 0:
                    logger.debug(f"{symbol} 资金不足, 无法买入")
                    order.status = OrderStatus.REJECTED
                    return None
                logger.debug(f"{symbol} 资金不足, 调整买入数量至 {order.quantity}")

        # 持仓检查(卖出)
        if order.direction == Direction.SELL:
            pos = self.positions.get(symbol, {})
            sellable = self._get_sellable_quantity(symbol, order.date)
            if sellable < order.quantity:
                order.quantity = round_lot(int(sellable))
                if order.quantity <= 0:
                    logger.debug(f"{symbol} 无可用持仓卖出")
                    order.status = OrderStatus.REJECTED
                    return None

        # 计算交易成本
        trade_amount = fill_price * order.quantity

        # 佣金
        commission = max(self.min_commission, trade_amount * self.commission_rate)
        # 印花税(仅卖出)
        stamp_duty = trade_amount * self.stamp_duty if order.direction == Direction.SELL else 0.0
        # 滑点成本
        if order.direction == Direction.BUY:
            slippage_cost = trade_amount * self.slippage
        else:
            slippage_cost = -trade_amount * self.slippage

        total_cost = commission + stamp_duty + abs(slippage_cost)

        return FillEvent(
            type=EventType.FILL,
            symbol=symbol,
            date=order.date,
            direction=order.direction,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            stamp_duty=stamp_duty,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            order_id=order.order_id,
        )

    def _determine_fill_price(self, order: OrderEvent, market: MarketEvent) -> float:
        """确定成交价 (考虑滑点)"""
        if order.order_type == OrderType.MARKET:
            # 市价买: 卖一价(≈current) + 滑点
            if order.direction == Direction.BUY:
                return market.close * (1 + self.slippage)
            else:
                return market.close * (1 - self.slippage)
        else:
            # 限价单: 仅当市价优于或等于限价时成交
            if order.direction == Direction.BUY:
                if market.close <= order.price:
                    return order.price
                else:
                    # 市价突破限价, 不成交
                    return market.close  # 简化处理: 按市价成交
            else:
                if market.close >= order.price:
                    return order.price
                else:
                    return market.close

    # ─── 成交应用 ───────────────────────────────────
    def _apply_fill(self, fill: FillEvent) -> None:
        """应用成交到资金和持仓"""
        symbol = fill.symbol

        if fill.direction == Direction.BUY:
            # 现金变化
            cost = fill.price * fill.quantity + fill.commission + fill.stamp_duty
            self.cash -= cost

            # 更新持仓
            if symbol in self.positions:
                pos = self.positions[symbol]
                old_qty = pos["quantity"]
                old_cost = pos["avg_cost"] * old_qty
                new_qty = old_qty + fill.quantity
                new_cost = (old_cost + fill.price * fill.quantity) / new_qty
                pos["quantity"] = new_qty
                pos["avg_cost"] = new_cost
            else:
                self.positions[symbol] = {
                    "quantity": fill.quantity,
                    "avg_cost": fill.price,
                    "market_value": fill.price * fill.quantity,
                }

            # 记录T+1
            self._today_buy[symbol] = self._today_buy.get(symbol, 0) + fill.quantity
            if symbol not in self._buy_records:
                self._buy_records[symbol] = []
            self._buy_records[symbol].append((fill.date, fill.quantity))

        else:
            # 卖出
            revenue = fill.price * fill.quantity - fill.commission - fill.stamp_duty
            self.cash += revenue

            # 更新持仓
            pos = self.positions.get(symbol)
            if pos:
                pos["quantity"] -= fill.quantity
                if pos["quantity"] <= 0:
                    del self.positions[symbol]

    # ─── T+1 可用持仓 ──────────────────────────────
    def _get_sellable_quantity(self, symbol: str, current_date: date) -> int:
        """计算T+1约束下可卖出数量"""
        pos = self.positions.get(symbol, {})
        total_qty = pos.get("quantity", 0)
        if total_qty <= 0:
            return 0

        # 计算今日及之后买入的数量(不可卖)
        locked = 0
        records = self._buy_records.get(symbol, [])
        for buy_dt, buy_qty in records:
            if buy_dt >= current_date:  # 当日买入不可卖
                locked += buy_qty

        return max(0, total_qty - locked)

    # ─── 日切处理 ───────────────────────────────────
    def end_of_day(self, current_date: date) -> None:
        """日终处理: 清理T+1记录中已可卖的"""
        # 清理 today_buy (T+1 之后可用)
        self._today_buy.clear()

        # 更新持仓市值
        for symbol, pos in self.positions.items():
            pos["market_value"] = pos["avg_cost"] * pos["quantity"]

    # ─── 组合快照 ───────────────────────────────────
    @property
    def market_value(self) -> float:
        """持仓总市值"""
        return sum(p.get("market_value", 0) for p in self.positions.values())

    @property
    def total_value(self) -> float:
        """总资产"""
        return self.cash + self.market_value

    @property
    def pnl(self) -> float:
        """累计盈亏"""
        return self.total_value - self.initial_cash
