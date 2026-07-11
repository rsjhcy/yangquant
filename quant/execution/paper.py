"""
模拟交易引擎
基于实时行情模拟真实交易执行
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger

from quant.backtest.broker import SimulatedBroker
from quant.backtest.events import Direction, OrderType
from quant.execution.base import ExecutionInterface


class PaperTradingEngine(ExecutionInterface):
    """模拟交易引擎

    用于在实盘数据上进行模拟交易:
    - 跟踪模拟账户的持仓和资金
    - 按市价匹配订单
    - 记录交易历史
    - 计算模拟PnL
    """

    name = "paper_trading"

    def __init__(
        self,
        initial_cash: float = 1_000_000,
        commission_rate: float = 0.00025,
        slippage: float = 0.0001,
    ):
        self.broker = SimulatedBroker(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage=slippage,
        )
        self._connected = False
        self._order_history: List[dict] = []
        self._pnl_history: List[dict] = []

    def connect(self) -> bool:
        self._connected = True
        logger.info("📝 模拟交易引擎已启动")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("📝 模拟交易引擎已停止")

    def submit_order(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: str = "LIMIT",
    ) -> Optional[str]:
        if not self._connected:
            logger.error("模拟交易引擎未连接")
            return None

        order_id = f"PAPER-{uuid4().hex[:8].upper()}"
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "status": "SUBMITTED",
            "create_time": datetime.now(),
        }
        self._order_history.append(order)
        logger.info(f"📝 模拟订单: {symbol} {direction} {quantity}股 @ {price}")
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        for order in self._order_history:
            if order["order_id"] == order_id and order["status"] == "SUBMITTED":
                order["status"] = "CANCELLED"
                return True
        return False

    def get_positions(self) -> Dict[str, dict]:
        return self.broker.positions

    def get_account(self) -> dict:
        return {
            "cash": self.broker.cash,
            "market_value": self.broker.market_value,
            "total_value": self.broker.total_value,
            "pnl": self.broker.pnl,
            "initial_cash": self.broker.initial_cash,
        }

    def get_orders(self, status: str = "all") -> List[dict]:
        if status == "all":
            return self._order_history
        return [o for o in self._order_history if o["status"] == status.upper()]

    def on_market_data(self, quotes: Dict[str, float]) -> None:
        """收到实时行情后更新持仓市值

        Args:
            quotes: {symbol: current_price}
        """
        for symbol, price in quotes.items():
            if symbol in self.broker.positions:
                pos = self.broker.positions[symbol]
                pos["market_value"] = pos["quantity"] * price

    def record_pnl(self, current_date: date) -> None:
        """记录当日PnL"""
        self._pnl_history.append({
            "date": current_date,
            "total_value": self.broker.total_value,
            "pnl": self.broker.pnl,
            "cash": self.broker.cash,
            "market_value": self.broker.market_value,
        })
