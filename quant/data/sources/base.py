"""
数据源抽象基类
定义统一的数据获取接口
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional

import pandas as pd


class DataSource(ABC):
    """数据源抽象基类 — 所有数据源必须实现此接口"""

    name: str = "base"

    @abstractmethod
    def get_daily(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        adjust: str = "qfq",  # qfq=前复权, hfq=后复权, none=不复权
    ) -> pd.DataFrame:
        """获取日线行情

        Returns:
            DataFrame with columns:
            - symbol: 股票代码
            - date: 日期
            - open, high, low, close: OHLC
            - volume: 成交量(股)
            - amount: 成交额(元)
            - turnover: 换手率(%)
            - adjust_factor: 复权因子(如有)
        """
        ...

    @abstractmethod
    def get_minute(
        self,
        symbols: List[str],
        trade_date: date,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """获取分钟线"""
        ...

    @abstractmethod
    def get_financials(
        self,
        symbols: List[str],
        report_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """获取财务数据"""
        ...

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """获取全A股列表"""
        ...

    @abstractmethod
    def get_index_daily(
        self,
        index_code: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """获取指数日线"""
        ...

    @abstractmethod
    def get_adjust_factor(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """获取复权因子"""
        ...
