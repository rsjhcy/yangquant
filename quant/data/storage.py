"""
本地数据存储
Parquet 分区存储: {root}/{data_type}/{symbol}/YYYY/MM/DD.parquet
支持增量更新、断点续传、数据校验
"""

from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from quant.config import config


class DataStorage:
    """本地 Parquet 分区存储引擎

    目录结构:
        data/
        ├── daily/
        │   ├── 000001/
        │   │   ├── 2024/
        │   │   │   ├── 01/
        │   │   │   │   ├── 02.parquet
        │   │   │   │   ├── 03.parquet
        │   │   │   │   └── ...
        │   │   │   └── 02/
        │   │   └── 2025/
        │   ├── 000002/
        │   └── ...
        ├── minute/
        ├── financials/
        └── index/
    """

    DATA_TYPES = ["daily", "minute", "financials", "index", "adjust_factor"]

    def __init__(self, root_dir: Optional[str] = None):
        self.root = Path(root_dir or config.data.root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        for dtype in self.DATA_TYPES:
            (self.root / dtype).mkdir(exist_ok=True)
        logger.info(f"💾 数据存储初始化: {self.root}")

    # ─── 路径工具 ─────────────────────────────────────
    def _daily_path(self, symbol: str, dt: date) -> Path:
        """日线数据存储路径"""
        return (
            self.root / "daily" / symbol / str(dt.year) / f"{dt.month:02d}" / f"{dt.day:02d}.parquet"
        )

    def _dtype_path(self, dtype: str, symbol: str, dt: date) -> Path:
        """通用数据存储路径"""
        return (
            self.root / dtype / symbol / str(dt.year) / f"{dt.month:02d}" / f"{dt.day:02d}.parquet"
        )

    # ─── 写入 ───────────────────────────────────────
    def save_daily(self, df: pd.DataFrame) -> int:
        """保存日线数据到分区 Parquet 文件

        Args:
            df: 日线 DataFrame, 必须含 symbol 和 date 列

        Returns:
            写入的文件数量
        """
        if df.empty:
            return 0

        count = 0
        for (symbol, dt), group in df.groupby(["symbol", "date"]):
            if isinstance(dt, datetime):
                dt = dt.date()
            elif isinstance(dt, str):
                dt = date.fromisoformat(dt)

            path = self._daily_path(str(symbol), dt)
            path.parent.mkdir(parents=True, exist_ok=True)
            # 不含 symbol 和 date 列（它们在路径中编码）
            data = group.drop(columns=["symbol", "date"], errors="ignore")
            data.to_parquet(path, index=False)
            count += 1

        logger.debug(f"💾 写入 {count} 个日线文件")
        return count

    def save_index(self, df: pd.DataFrame) -> int:
        """保存指数行情"""
        if df.empty:
            return 0
        count = 0
        for dt, group in df.groupby("date"):
            if isinstance(dt, datetime):
                dt = dt.date()
            path = self._dtype_path("index", "index", dt)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = group.drop(columns=["date"], errors="ignore")
            # 合并已有数据
            if path.exists():
                existing = pd.read_parquet(path)
                data = pd.concat([existing, data]).drop_duplicates()
            data.to_parquet(path, index=False)
            count += 1
        return count

    # ─── 读取 ───────────────────────────────────────
    def load_daily(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """读取日线数据

        Args:
            symbols: 股票代码列表
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            DataFrame with columns: symbol, date, open, close, high, low, volume, amount, turnover
        """
        frames = []
        for symbol in symbols:
            symbol_dir = self.root / "daily" / symbol
            if not symbol_dir.exists():
                logger.debug(f"符号 {symbol} 无本地数据")
                continue

            for year_dir in sorted(symbol_dir.iterdir()):
                if not year_dir.is_dir():
                    continue
                try:
                    year = int(year_dir.name)
                except ValueError:
                    continue

                if year < start_date.year or year > end_date.year:
                    continue

                for month_dir in sorted(year_dir.iterdir()):
                    if not month_dir.is_dir():
                        continue
                    for fpath in sorted(month_dir.glob("*.parquet")):
                        try:
                            file_date = date(
                                int(year_dir.name),
                                int(month_dir.name),
                                int(fpath.stem),
                            )
                        except ValueError:
                            continue

                        if file_date < start_date or file_date > end_date:
                            continue

                        try:
                            df = pd.read_parquet(fpath)
                            df["symbol"] = symbol
                            df["date"] = file_date
                            frames.append(df)
                        except Exception as e:
                            logger.warning(f"读取 {fpath} 失败: {e}")

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        # 重新排列列顺序
        col_order = ["symbol", "date", "open", "high", "low", "close", "volume", "amount", "turnover", "amplitude", "pct_chg"]
        available = [c for c in col_order if c in result.columns]
        remaining = [c for c in result.columns if c not in available]
        return result[available + remaining]

    def load_index(
        self,
        index_code: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """读取指数行情"""
        index_dir = self.root / "index"
        if not index_dir.exists():
            return pd.DataFrame()

        frames = []
        for year_dir in sorted(index_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            if year < start_date.year or year > end_date.year:
                continue

            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for fpath in sorted(month_dir.glob("*.parquet")):
                    try:
                        file_date = date(int(year_dir.name), int(month_dir.name), int(fpath.stem))
                    except ValueError:
                        continue
                    if file_date < start_date or file_date > end_date:
                        continue
                    try:
                        df = pd.read_parquet(fpath)
                        if "index_code" in df.columns:
                            df = df[df["index_code"] == index_code]
                        if not df.empty:
                            df["date"] = file_date
                            frames.append(df)
                    except Exception as e:
                        logger.warning(f"读取指数 {fpath} 失败: {e}")

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ─── 增量更新 ───────────────────────────────────
    def get_latest_date(self, symbol: str, dtype: str = "daily") -> Optional[date]:
        """获取某只股票的最新数据日期"""
        dtype_dir = self.root / dtype / symbol
        if not dtype_dir.exists():
            return None

        all_files = list(dtype_dir.rglob("*.parquet"))
        if not all_files:
            return None

        # 按路径解析日期，返回最新的
        latest = None
        for f in all_files:
            # 路径格式: .../symbol/YYYY/MM/DD.parquet
            parts = f.parts
            try:
                d = date(int(parts[-3]), int(parts[-2]), int(f.stem))
                if latest is None or d > latest:
                    latest = d
            except (ValueError, IndexError):
                pass
        return latest

    def get_coverage(
        self, symbol: str, dtype: str = "daily"
    ) -> Dict[str, List[date]]:
        """获取某只股票的数据覆盖情况 (有数据的日期列表 + 缺失区间)"""
        dtype_dir = self.root / dtype / symbol
        if not dtype_dir.exists():
            return {"dates": [], "missing_ranges": [(None, None)]}

        dates = []
        for f in sorted(dtype_dir.rglob("*.parquet")):
            try:
                parts = f.parts
                d = date(int(parts[-3]), int(parts[-2]), int(f.stem))
                dates.append(d)
            except (ValueError, IndexError):
                pass

        dates = sorted(set(dates))
        return {"dates": dates, "count": len(dates)}
