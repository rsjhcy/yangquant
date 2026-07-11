"""
执行接口抽象
定义实盘执行的统一接口
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Dict, List, Optional


class ExecutionInterface(ABC):
    """执行接口 — 模拟交易和实盘交易必须实现此接口"""

    name: str = "execution_base"

    @abstractmethod
    def connect(self) -> bool:
        """连接交易系统"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        ...

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        direction: str,     # 'BUY' | 'SELL'
        quantity: int,
        price: Optional[float] = None,
        order_type: str = "LIMIT",
    ) -> Optional[str]:
        """提交订单 → 返回订单ID"""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        ...

    @abstractmethod
    def get_positions(self) -> Dict[str, dict]:
        """获取当前持仓"""
        ...

    @abstractmethod
    def get_account(self) -> dict:
        """获取账户信息"""
        ...

    @abstractmethod
    def get_orders(self, status: str = "all") -> List[dict]:
        """获取订单列表"""
        ...
