"""
A股交易约束
涨跌停限制 / T+1 / 最小交易单位 / 停牌处理
"""

from datetime import date
from typing import Optional


def round_lot(quantity: int) -> int:
    """规整为手数 (100股的整数倍)"""
    return max(0, (quantity // 100) * 100)


def calc_limit_price(close: float, direction: str = "up", is_st: bool = False) -> float:
    """计算涨跌停价格

    Args:
        close: 前收盘价
        direction: 'up' 涨停价 / 'down' 跌停价
        is_st: 是否为ST股票

    Returns:
        涨跌停价格
    """
    if close <= 0:
        return 0.0

    rate = 0.05 if is_st else 0.10

    if direction == "up":
        price = close * (1 + rate)
    else:
        price = close * (1 - rate)

    # 四舍五入到分
    return round(price, 2)


def is_limit_hit(price: float, pre_close: float, direction: str, is_st: bool = False) -> bool:
    """判断是否触及涨跌停"""
    limit = calc_limit_price(pre_close, direction, is_st)
    if direction == "up":
        return price >= limit - 1e-6
    else:
        return price <= limit + 1e-6


def is_t_plus_one_available(
    buy_date: date,
    sell_date: date,
    buy_quantity: int,
    sold_quantity: int,
) -> int:
    """T+1检查: 返回可卖出的数量

    Args:
        buy_date: 买入日期
        sell_date: 当日日期
        buy_quantity: 买入数量
        sold_quantity: 已卖出数量

    Returns:
        可卖出数量
    """
    if sell_date <= buy_date:
        return 0
    return max(0, buy_quantity - sold_quantity)


class AShareConstraints:
    """A股交易约束管理器"""

    COMMISSION_RATE = 0.00025        # 默认佣金万2.5
    MIN_COMMISSION = 5.0             # 最低佣金5元
    STAMP_DUTY_RATE = 0.001          # 印花税（仅卖出）
    TRANSFER_FEE_RATE = 0.00002      # 过户费万分之0.2
    MIN_LOT = 100                    # 最小交易单位1手

    @staticmethod
    def calc_commission(amount: float, rate: Optional[float] = None) -> float:
        """计算佣金"""
        r = rate or AShareConstraints.COMMISSION_RATE
        return max(AShareConstraints.MIN_COMMISSION, amount * r)

    @staticmethod
    def calc_stamp_duty(amount: float) -> float:
        """计算印花税(卖出)"""
        return amount * AShareConstraints.STAMP_DUTY_RATE

    @staticmethod
    def calc_transfer_fee(amount: float) -> float:
        """计算过户费"""
        return amount * AShareConstraints.TRANSFER_FEE_RATE

    @staticmethod
    def calc_total_cost(
        amount: float,
        direction: str,  # 'buy' or 'sell'
        commission_rate: Optional[float] = None,
    ) -> dict:
        """计算总交易成本"""
        commission = AShareConstraints.calc_commission(amount, commission_rate)
        transfer = AShareConstraints.calc_transfer_fee(amount)

        if direction == "sell":
            stamp = AShareConstraints.calc_stamp_duty(amount)
        else:
            stamp = 0.0

        return {
            "commission": commission,
            "stamp_duty": stamp,
            "transfer_fee": transfer,
            "total": commission + stamp + transfer,
        }

    @staticmethod
    def calc_max_buy_quantity(
        price: float,
        available_cash: float,
        position_pct_limit: float = 0.20,
        total_value: float = 0,
    ) -> int:
        """计算最大可买股数（考虑资金和仓位比例）"""
        if price <= 0:
            return 0

        # 资金约束
        max_by_cash = int(available_cash / price)

        # 仓位约束
        if total_value > 0 and position_pct_limit < 1.0:
            max_by_position = int(total_value * position_pct_limit / price)
        else:
            max_by_position = max_by_cash

        max_qty = min(max_by_cash, max_by_position)
        return round_lot(max_qty)
