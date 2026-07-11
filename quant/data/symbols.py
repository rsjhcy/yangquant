"""
股票池管理
管理股票列表、指数成分、行业分类等
"""

from datetime import date
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


class SymbolManager:
    """股票池管理器

    功能:
    - 全A股列表
    - 指数成分股 (沪深300/中证500/中证1000/创业板指...)
    - 行业分类 (申万一级)
    - 股票代码标准化
    """

    _instance: Optional["SymbolManager"] = None

    # 常见指数代码映射
    _INDEX_MAP = {
        "000300": "沪深300",
        "000905": "中证500",
        "000852": "中证1000",
        "000016": "上证50",
        "399006": "创业板指",
        "399005": "中小100",
        "000688": "科创50",
        "399330": "沪深300成长",
    }

    def __new__(cls) -> "SymbolManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._stock_list = pd.DataFrame()
            cls._instance._index_members: Dict[str, pd.DataFrame] = {}
        return cls._instance

    # ─── 股票列表 ────────────────────────────────────
    def load_stock_list(self, force: bool = False) -> pd.DataFrame:
        """加载全A股列表"""
        if not self._stock_list.empty and not force:
            return self._stock_list

        from quant.data.sources.akshare_ import AkshareSource

        source = AkshareSource()
        self._stock_list = source.get_stock_list()
        logger.info(f"📋 股票列表加载完成: {len(self._stock_list)} 只")
        return self._stock_list

    @property
    def all_stocks(self) -> pd.DataFrame:
        """获取全A股列表"""
        if self._stock_list.empty:
            self.load_stock_list()
        return self._stock_list

    def filter_by_market(self, market: str = "all") -> pd.DataFrame:
        """按板块过滤
        Args:
            market: 'sh'=沪市, 'sz'=深市, 'all'=全部
        """
        df = self.all_stocks
        if market == "sh":
            return df[df["symbol"].str.startswith("6")]
        elif market == "sz":
            return df[~df["symbol"].str.startswith("6")]
        return df

    def get_industry_groups(self) -> Dict[str, List[str]]:
        """获取行业分类 — 返回 {行业名: [股票代码...]}"""
        df = self.all_stocks
        if df.empty or "industry" not in df.columns:
            return {}

        grouped = df.groupby("industry")["symbol"].apply(list).to_dict()
        return grouped

    # ─── 指数成分 ────────────────────────────────────
    def load_index_members(self, index_code: str) -> List[str]:
        """加载指数成分股"""
        if index_code not in self._INDEX_MAP:
            logger.warning(f"未知指数: {index_code}, 尝试直接获取")

        try:
            import akshare as ak

            raw = ak.index_stock_cons(index_code)
            if raw is not None and not raw.empty:
                # akshare 不同指数返回的列名可能不同
                symbol_col = (
                    "品种代码" if "品种代码" in raw.columns
                    else "成分券代码" if "成分券代码" in raw.columns
                    else raw.columns[0]
                )
                symbols = raw[symbol_col].astype(str).tolist()
                logger.info(f"📊 {self._INDEX_MAP.get(index_code, index_code)} 成分股: {len(symbols)} 只")
                return symbols
        except Exception as e:
            logger.error(f"获取指数成分股失败 ({index_code}): {e}")

        return []

    @lru_cache(maxsize=16)
    def get_index_members_cached(self, index_code: str) -> List[str]:
        """获取指数成分（缓存）"""
        return self.load_index_members(index_code)

    # ─── 股票代码标准化 ──────────────────────────────
    @staticmethod
    def normalize(symbol: str) -> str:
        """标准化股票代码为6位字符串"""
        s = str(symbol).strip().upper().replace(".SH", "").replace(".SZ", "")
        return s.zfill(6)

    @staticmethod
    def to_tushare_format(symbol: str) -> str:
        """转成 tushare 格式: 000001.SZ / 600519.SH"""
        s = SymbolManager.normalize(symbol)
        market = "SH" if s.startswith("6") else "SZ"
        return f"{s}.{market}"

    @staticmethod
    def get_market(symbol: str) -> str:
        """判断市场: SH 或 SZ"""
        s = SymbolManager.normalize(symbol)
        return "SH" if s.startswith("6") else "SZ"
