"""
A股交易日历
处理交易日判断、节假日、T+N计算等
"""

from datetime import date, timedelta
from functools import lru_cache
from typing import List, Optional

import pandas as pd
from loguru import logger


class TradingCalendar:
    """A股交易日历

    特性:
    - 判断某日是否为交易日
    - 获取 N 个交易日前的日期
    - 获取两个日期之间的交易日列表
    - 自动缓存交易日历
    """

    _instance: Optional["TradingCalendar"] = None
    _trading_days: List[date] = []
    _trading_set: set = set()
    _loaded: bool = False

    # A股主要节假日（按年维护）
    _FIXED_HOLIDAYS = {
        # 春节（每年需更新）
        2024: [
            date(2024, 2, 9), date(2024, 2, 10), date(2024, 2, 11),
            date(2024, 2, 12), date(2024, 2, 13), date(2024, 2, 14),
            date(2024, 2, 15), date(2024, 2, 16), date(2024, 2, 17),
        ],
        2025: [
            date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30),
            date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 2),
            date(2025, 2, 3), date(2025, 2, 4),
        ],
        2026: [
            date(2026, 2, 17), date(2026, 2, 18), date(2026, 2, 19),
            date(2026, 2, 20), date(2026, 2, 21), date(2026, 2, 22),
            date(2026, 2, 23),
        ],
    }
    # 其他休市规则（元旦/清明/五一/端午/中秋/国庆）每年有细微差异
    # 此处提供基础规则，完整版本建议从 akshare 获取交易日历

    def __new__(cls) -> "TradingCalendar":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, start_year: int = 2010, end_year: int = 2030) -> None:
        """生成交易日历

        基于规则: 周一到周五 - 节假日
        """
        if self._loaded:
            return

        start = date(start_year, 1, 1)
        end = date(end_year, 12, 31)

        # 生成所有工作日
        days = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # 周一到周五
                days.append(current)
            current += timedelta(days=1)

        # 排除已知节假日
        all_holidays = set()
        for year_holidays in self._FIXED_HOLIDAYS.values():
            all_holidays.update(year_holidays)

        self._trading_days = [d for d in days if d not in all_holidays]
        self._trading_days.sort()
        self._trading_set = set(self._trading_days)
        self._loaded = True
        logger.info(f"📅 交易日历加载完成: {len(self._trading_days)} 个交易日")

    def is_trading_day(self, d: date) -> bool:
        """判断是否为交易日"""
        if not self._loaded:
            self.load()
        return d in self._trading_set

    def next_trading_day(self, d: date, offset: int = 1) -> date:
        """获取 N 个交易日之后的日期"""
        if not self._loaded:
            self.load()
        if offset <= 0:
            return d
        count = 0
        current = d + timedelta(days=1)
        while True:
            if self.is_trading_day(current):
                count += 1
                if count >= offset:
                    return current
            current += timedelta(days=1)

    def prev_trading_day(self, d: date, offset: int = 1) -> date:
        """获取 N 个交易日之前的日期"""
        if not self._loaded:
            self.load()
        if offset <= 0:
            return d
        count = 0
        current = d - timedelta(days=1)
        while True:
            if self.is_trading_day(current):
                count += 1
                if count >= offset:
                    return current
            current -= timedelta(days=1)

    def get_trading_days(self, start: date, end: date) -> List[date]:
        """获取起止日期之间的所有交易日"""
        if not self._loaded:
            self.load()
        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)
        return [d for d in self._trading_days if start <= d <= end]

    def count_trading_days(self, start: date, end: date) -> int:
        """计算起止日期之间的交易日数量"""
        return len(self.get_trading_days(start, end))

    def most_recent_trading_day(self, d: Optional[date] = None) -> date:
        """获取最近的交易日（含当日）"""
        if d is None:
            from datetime import date as _date
            d = _date.today()
        if not self._loaded:
            self.load()
        while d not in self._trading_set:
            d -= timedelta(days=1)
        return d


# 全局实例
cal = TradingCalendar()
