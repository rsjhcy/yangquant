"""
回测引擎 - 事件系统
定义回测过程中的所有事件类型
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional


class EventType(Enum):
    """事件类型"""
    MARKET = "MARKET"            # 行情事件
    SIGNAL = "SIGNAL"            # 信号事件
    ORDER = "ORDER"              # 订单事件
    FILL = "FILL"                # 成交事件
    PORTFOLIO = "PORTFOLIO"      # 组合更新事件
    SETTLEMENT = "SETTLEMENT"    # 日结事件


class OrderType(Enum):
    """订单类型"""
    MARKET = "MARKET"            # 市价单
    LIMIT = "LIMIT"              # 限价单


class OrderStatus(Enum):
    """订单状态"""
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Direction(Enum):
    """买卖方向"""
    BUY = "BUY"                  # 买入
    SELL = "SELL"                # 卖出


@dataclass
class MarketEvent:
    """行情事件 — 推动回测循环的核心事件"""
    type: EventType = EventType.MARKET
    symbol: str = ""
    date: date = field(default_factory=date.today)
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    amount: float = 0.0
    turnover: float = 0.0
    pre_close: float = 0.0          # 前收盘价 (用于计算涨跌停)
    is_suspended: bool = False      # 是否停牌
    is_limit_up: bool = False       # 涨停
    is_limit_down: bool = False     # 跌停


@dataclass
class SignalEvent:
    """信号事件 — 策略产生的交易信号"""
    type: EventType = EventType.SIGNAL
    symbol: str = ""
    date: date = field(default_factory=date.today)
    direction: Direction = Direction.BUY
    strength: float = 1.0           # 信号强度 [0, 1]
    reason: str = ""                # 信号原因
    target_weight: float = 0.0      # 目标权重 (用于组合优化)


@dataclass
class OrderEvent:
    """订单事件 — 发送到模拟券商的订单"""
    type: EventType = EventType.ORDER
    symbol: str = ""
    date: date = field(default_factory=date.today)
    direction: Direction = Direction.BUY
    order_type: OrderType = OrderType.LIMIT
    quantity: int = 0               # 股数 (100的整数倍)
    price: float = 0.0              # 委托价格
    order_id: str = ""
    status: OrderStatus = OrderStatus.CREATED
    reason: str = ""


@dataclass
class FillEvent:
    """成交事件 — 订单成交回报"""
    type: EventType = EventType.FILL
    symbol: str = ""
    date: date = field(default_factory=date.today)
    direction: Direction = Direction.BUY
    quantity: int = 0
    price: float = 0.0              # 成交价
    commission: float = 0.0         # 佣金
    stamp_duty: float = 0.0         # 印花税
    slippage_cost: float = 0.0      # 滑点成本
    total_cost: float = 0.0         # 总成本 (含佣金印花税)
    order_id: str = ""


@dataclass
class PortfolioEvent:
    """组合更新事件 — 持仓/现金/pnl变化"""
    type: EventType = EventType.PORTFOLIO
    date: date = field(default_factory=date.today)
    cash: float = 0.0
    market_value: float = 0.0       # 持仓市值
    total_value: float = 0.0        # 总资产 = 现金 + 市值
    pnl: float = 0.0                # 当日盈亏
    cumulative_pnl: float = 0.0     # 累计盈亏
    cumulative_return: float = 0.0  # 累计收益率
    positions: dict = field(default_factory=dict)  # {symbol: {quantity, avg_cost, market_value}}


@dataclass
class SettlementEvent:
    """日结事件 — 每日收盘后处理"""
    type: EventType = EventType.SETTLEMENT
    date: date = field(default_factory=date.today)
