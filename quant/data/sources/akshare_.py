"""
akshare 数据源适配器
基于 akshare (免费A股数据接口) 实现 DataSource 接口
"""

import time
import random
from datetime import date, timedelta
from typing import Dict, List, Optional

import akshare as ak
import pandas as pd
from loguru import logger

from quant.data.sources.base import DataSource
from quant.utils.decorators import timer


class AkshareSource(DataSource):
    """akshare 数据源 — 免费A股数据主力

    支持:
    - 日线行情 (前/后复权)
    - 分钟线 (5/15/30/60分钟)
    - 财务数据 (季报)
    - 复权因子
    - 指数行情
    - 全A股列表

    注意:
    - akshare 后端对接东方财富等公开数据源
    - 高频请求可能触发反爬，已内置重试+随机延迟
    """

    name = "akshare"
    # 列名映射: akshare → 标准列名
    _DAILY_COLUMN_MAP = {
        "股票代码": "symbol",
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    }

    def __init__(self):
        self._setup_session()
        logger.info("akshare 数据源已初始化")

    def _setup_session(self):
        """初始化请求配置"""
        pass  # akshare 自行管理 session，不需要额外配置

    # ─── 日线行情 ─────────────────────────────────
    @timer(label="akshare.get_daily")
    def get_daily(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取日线行情

        Args:
            symbols: 股票代码列表, 如 ['000001', '600519']
            start_date: 起始日期
            end_date: 结束日期
            adjust: 复权方式 - 'qfq'(前复权) / 'hfq'(后复权) / ''(不复权)

        Returns:
            标准化DataFrame, 包含 OHLCV + 换手率
        """
        frames = []
        failed = []

        for i, symbol in enumerate(symbols):
            df = self._get_single_daily_with_retry(symbol, start_date, end_date, adjust)
            if not df.empty:
                frames.append(df)
            else:
                failed.append(symbol)

            # 多只股票之间加随机延迟，避免触发反爬
            if i < len(symbols) - 1:
                delay = random.uniform(1.5, 3.0)
                time.sleep(delay)
            elif len(symbols) == 1:
                # 单只股票也加一个短延迟，平滑请求频率
                time.sleep(random.uniform(0.5, 1.0))

        if failed:
            logger.warning(
                f"[{len(failed)}/{len(symbols)}] 只股票获取失败: {failed}"
                f" (akshare 后端数据源可能临时限流，稍后重试)"
            )

        if not frames:
            logger.error(f"所有股票均获取失败: {symbols}")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        logger.info(
            f"日线数据: {len(symbols)} 只请求, "
            f"{result['symbol'].nunique()} 只成功, "
            f"{result['date'].nunique()} 个交易日"
        )
        return result

    def _get_single_daily_with_retry(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        adjust: str,
        max_retries: int = 4,
    ) -> pd.DataFrame:
        """获取单只股票日线 (带重试 + 备用端点)

        重试策略:
        - 第1-2次: 使用主端点 ak.stock_zh_a_hist
        - 第3-4次: 切换到备用端点 ak.stock_zh_a_hist_em
        - 每次重试的等待时间指数增长，加随机抖动
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # 第1-2次用主端点，之后切换备用端点
                use_fallback = attempt > 2
                df = self._get_single_daily(
                    symbol, start_date, end_date, adjust,
                    use_fallback=use_fallback,
                )
                if not df.empty:
                    if use_fallback:
                        logger.debug(f"{symbol}: 备用端点成功")
                    return df
                # 空数据 — 可能是新股/停牌/代码无效，不重试
                if attempt == 1:
                    logger.debug(f"{symbol}: 返回空数据 (可能停牌或代码无效)")
                return pd.DataFrame()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # 指数退避: 3s, 6s, 12s + 随机抖动
                    wait = 3.0 * (2.0 ** (attempt - 1)) + random.uniform(0.5, 2.0)
                    endpoint = "备用" if attempt >= 2 else "主"
                    logger.debug(
                        f"{symbol}: #{attempt}失败 ({type(e).__name__}), "
                        f"{wait:.1f}s后重试({endpoint}端点)..."
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        f"{symbol}: {max_retries}次重试全部失败 "
                        f"({type(e).__name__})"
                    )

        return pd.DataFrame()

    def _to_tx_symbol(self, symbol: str) -> str:
        """转成腾讯接口格式: 000001 -> sz000001, 600519 -> sh600519"""
        s = symbol.strip()
        if s.startswith(("0", "3")):
            return f"sz{s}"
        return f"sh{s}"

    def _normalize_tx_columns(self, raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """标准化腾讯接口返回的列名

        腾讯接口返回: date, open, close, high, low, amount (成交额/元)
        需要补全: symbol, volume, turnover
        """
        df = raw.copy()
        df["symbol"] = symbol

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date

        # 腾讯接口没有 volume，设为0 (不影响回测核心逻辑)
        if "volume" not in df.columns:
            df["volume"] = 0

        # amount 在腾讯接口返回的是成交额(元)，保持原样
        if "turnover" not in df.columns:
            df["turnover"] = 0.0

        for col in ["open", "close", "high", "low", "volume", "amount", "turnover"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["symbol", "date", "open", "high", "low", "close", "volume", "amount", "turnover"]]

    def _get_single_daily(
        self, symbol: str, start_date: date, end_date: date, adjust: str,
        use_fallback: bool = False,
    ) -> pd.DataFrame:
        """获取单只股票日线 (单次请求)

        Args:
            use_fallback: True=腾讯接口, False=东方财富接口
        """
        if use_fallback:
            # 备用: 腾讯历史行情 (完全不同的后端服务器)
            tx_sym = self._to_tx_symbol(symbol)
            raw = ak.stock_zh_a_hist_tx(
                symbol=tx_sym,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                adjust=adjust or "",
            )
            if raw is not None and not raw.empty:
                return self._normalize_tx_columns(raw, symbol)
            return pd.DataFrame()

        # 主端点: 东方财富
        raw = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            adjust=adjust,
        )

        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.rename(columns=self._DAILY_COLUMN_MAP)
        available_cols = [c for c in self._DAILY_COLUMN_MAP.values() if c in df.columns]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date

        for col in ["open", "close", "high", "low", "volume", "amount", "turnover"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[available_cols]

    # ─── 分钟线 ─────────────────────────────────
    def get_minute(
        self,
        symbols: List[str],
        trade_date: date,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """获取分钟线"""
        frames = []
        period_map = {"1min": "1", "5min": "5", "15min": "15", "30min": "30", "60min": "60"}
        period = period_map.get(freq, "5")

        for symbol in symbols:
            try:
                raw = ak.stock_zh_a_hist_min_em(
                    symbol=symbol,
                    period=period,
                    start_date=f"{trade_date:%Y-%m-%d} 09:30:00",
                    end_date=f"{trade_date:%Y-%m-%d} 15:00:00",
                )
                if raw is not None and not raw.empty:
                    df = raw.rename(columns={
                        "时间": "datetime",
                        "开盘": "open",
                        "收盘": "close",
                        "最高": "high",
                        "最低": "low",
                        "成交量": "volume",
                        "成交额": "amount",
                    })
                    df["symbol"] = symbol
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    frames.append(df)
            except Exception as e:
                logger.warning(f"获取 {symbol} 分钟线失败: {e}")

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ─── 财务数据 ─────────────────────────────────
    def get_financials(
        self,
        symbols: List[str],
        report_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """获取资产负债表/利润表/现金流量表"""
        logger.warning("akshare 批量获取财务数据速度较慢，建议使用 symbol_data.py 缓存后的数据")
        frames = []
        for symbol in symbols:
            try:
                # 获取主要财务指标
                raw = ak.stock_financial_abstract(symbol=symbol)
                if raw is not None and not raw.empty:
                    df = raw.copy()
                    df["symbol"] = symbol
                    frames.append(df)
            except Exception as e:
                logger.warning(f"获取 {symbol} 财务数据失败: {e}")

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ─── 股票列表 ─────────────────────────────────
    def get_stock_list(self) -> pd.DataFrame:
        """获取全A股股票列表"""
        try:
            raw = ak.stock_zh_a_spot_em()
            df = pd.DataFrame({
                "symbol": raw["代码"],
                "name": raw["名称"],
                "industry": raw.get("所属行业", ""),
                "market": raw["代码"].apply(lambda x: "SH" if x.startswith("6") else "SZ"),
                "list_date": pd.NaT,
            })
            logger.info(f"📋 获取到 {len(df)} 只A股")
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()

    # ─── 指数行情 ─────────────────────────────────
    def get_index_daily(
        self,
        index_code: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """获取指数日线行情"""
        index_map = {
            "000001": "sh000001",   # 上证指数
            "000300": "sh000300",   # 沪深300
            "000905": "sh000905",   # 中证500
            "000852": "sh000852",   # 中证1000
            "399001": "sz399001",   # 深证成指
            "399006": "sz399006",   # 创业板指
            "000016": "sh000016",   # 上证50
            "399673": "sz399673",   # 创业板50
        }

        symbol = index_map.get(index_code, f"sh{index_code}")
        try:
            raw = ak.stock_zh_index_daily_em(symbol=symbol)
            if raw is None or raw.empty:
                return pd.DataFrame()

            df = raw.rename(columns={
                "date": "date",
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "volume": "volume",
                "amount": "amount",
            })
            df["date"] = pd.to_datetime(df["date"]).dt.date

            if "close" in df.columns:
                numeric_cols = ["open", "close", "high", "low", "volume", "amount"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            return df[mask].copy()
        except Exception as e:
            logger.error(f"获取指数 {index_code} 行情失败: {e}")
            return pd.DataFrame()

    # ─── 复权因子 ─────────────────────────────────
    def get_adjust_factor(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """获取复权因子 (通过比较不复权收���与前复权收盘推算)"""
        # akshare 不直接提供复权因子，通过 qfq/hfq 数据反推
        try:
            df_qfq = self.get_daily(symbols, start_date, end_date, adjust="qfq")
            df_none = self.get_daily(symbols, start_date, end_date, adjust="")

            if df_qfq.empty or df_none.empty:
                return pd.DataFrame()

            merged = df_qfq.merge(
                df_none[["symbol", "date", "close"]],
                on=["symbol", "date"],
                suffixes=("_qfq", ""),
            )
            merged["adjust_factor"] = merged["close_qfq"] / merged["close"]
            return merged[["symbol", "date", "adjust_factor"]]
        except Exception as e:
            logger.error(f"计算复权因子失败: {e}")
            return pd.DataFrame()
