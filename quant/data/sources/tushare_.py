"""
tushare 数据源适配器 (备用)
需要 tushare token，免费注册: https://tushare.pro

相比 akshare 的优势:
- 更稳定的 API
- 更完整的基本面数据
- 分钟线支持更好
"""

from datetime import date
from typing import List, Optional

import pandas as pd
from loguru import logger

from quant.data.sources.base import DataSource
from quant.utils.decorators import retry


class TushareSource(DataSource):
    """tushare 数据源 — A股备用数据接口

    使用前:
        1. pip install tushare
        2. 注册获取 token: https://tushare.pro
        3. 在 config.yaml 中设置 sources.tushare_token
    """

    name = "tushare"

    def __init__(self, token: str = ""):
        try:
            import tushare as ts
        except ImportError:
            raise ImportError("请安装 tushare: pip install tushare")

        from quant.config import config

        self.token = token or config.sources.tushare_token
        if not self.token:
            logger.warning("tushare token 未设置，部分接口可能受限")

        ts.set_token(self.token)
        self._pro = ts.pro_api()
        logger.info("🔌 tushare 数据源已初始化")

    @retry(max_attempts=3, delay=1.0)
    def get_daily(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取日线行情

        tushare 复权字段:
            qfq → 前复权
            hfq → 后复权
            ''  → 不复权
        """
        try:
            ts_codes = [self._to_ts_code(s) for s in symbols]
            ts_code_str = ",".join(ts_codes)

            df = self._pro.daily(
                ts_code=ts_code_str,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # 如果请求复权数据
            if adjust in ("qfq", "hfq"):
                adj_factor = self._pro.adj_factor(
                    ts_code=ts_code_str,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                )
                if adj_factor is not None and not adj_factor.empty:
                    f = "qfq_factor" if adjust == "qfq" else "hfq_factor"
                    df = df.merge(
                        adj_factor[["ts_code", "trade_date", f]],
                        on=["ts_code", "trade_date"],
                        how="left",
                    )
                    for col in ["open", "high", "low", "close"]:
                        df[col] = df[col] * df[f]

            # 标准化列名
            result = df.rename(columns={
                "ts_code": "symbol",
                "trade_date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume",
                "amount": "amount",
            })

            result["symbol"] = result["symbol"].str.replace(".SH", "").str.replace(".SZ", "")
            result["date"] = pd.to_datetime(result["date"]).dt.date

            return result[["symbol", "date", "open", "high", "low", "close", "volume", "amount"]]

        except Exception as e:
            logger.error(f"tushare 获取日线失败: {e}")
            return pd.DataFrame()

    def get_minute(
        self,
        symbols: List[str],
        trade_date: date,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """获取分钟线 (tushare 分钟线需要较高积分)"""
        logger.warning("tushare 分钟线接口需要较高权限积分")
        return pd.DataFrame()

    def get_financials(
        self,
        symbols: List[str],
        report_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """获取财务数据"""
        try:
            ts_codes = [self._to_ts_code(s) for s in symbols]
            ts_code_str = ",".join(ts_codes)

            # 获取三大报表关键指标
            income = self._pro.income(
                ts_code=ts_code_str,
                period=report_date.strftime("%Y%m%d") if report_date else None,
                fields="ts_code,end_date,revenue,total_cogs,n_income,roe,basic_eps",
            )

            balance = self._pro.balancesheet(
                ts_code=ts_code_str,
                period=report_date.strftime("%Y%m%d") if report_date else None,
                fields="ts_code,end_date,total_assets,total_liab,total_hldr_eqy",
            )

            if income is not None and balance is not None:
                result = income.merge(balance, on=["ts_code", "end_date"])
                result["symbol"] = result["ts_code"].str.replace(".SH", "").str.replace(".SZ", "")
                return result

            return pd.DataFrame()
        except Exception as e:
            logger.error(f"tushare 获取财务数据失败: {e}")
            return pd.DataFrame()

    def get_stock_list(self) -> pd.DataFrame:
        """获取全A股列表"""
        try:
            df = self._pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,list_date",
            )
            if df is not None:
                df["symbol"] = df["ts_code"].str.replace(".SH", "").str.replace(".SZ", "")
                df["list_date"] = pd.to_datetime(df["list_date"]).dt.date
                return df.rename(columns={"industry": "industry"})
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"tushare 获取股票列表失败: {e}")
            return pd.DataFrame()

    def get_index_daily(
        self,
        index_code: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """获取指数日线"""
        try:
            # 转成 tushare 格式
            ts_code = self._to_ts_code(index_code, is_index=True)

            df = self._pro.index_daily(
                ts_code=ts_code,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
            )
            if df is not None and not df.empty:
                return df.rename(columns={
                    "ts_code": "index_code",
                    "trade_date": "date",
                })
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"tushare 获取指数行情失败: {e}")
            return pd.DataFrame()

    def get_adjust_factor(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """获取复权因子"""
        try:
            ts_codes = [self._to_ts_code(s) for s in symbols]
            ts_code_str = ",".join(ts_codes)

            df = self._pro.adj_factor(
                ts_code=ts_code_str,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
            )
            if df is not None:
                df["symbol"] = df["ts_code"].str.replace(".SH", "").str.replace(".SZ", "")
                df["date"] = pd.to_datetime(df["trade_date"]).dt.date
                df["adjust_factor"] = df["adj_factor"]
                return df[["symbol", "date", "adjust_factor"]]
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"tushare 获取复权因子失败: {e}")
            return pd.DataFrame()

    @staticmethod
    def _to_ts_code(symbol: str, is_index: bool = False) -> str:
        """标准化代码为 tushare 格式"""
        s = str(symbol).strip().upper()
        if is_index:
            return f"{s}.SH" if s.startswith("0") else f"{s}.SZ"
        market = "SH" if s.startswith("6") else "SZ"
        return f"{s}.{market}"
