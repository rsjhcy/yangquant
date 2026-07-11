"""
数据自动更新器
支持增量更新、定时任务、股票池管理
"""

import json
import time
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from quant.data.sources import AkshareSource
from quant.data.storage import DataStorage
from quant.config import config


class DataUpdater:
    """数据自动更新器

    功能:
    - 增量更新: 只下载本地缺失的日期
    - 股票池: 维护一个关注列表
    - 定时任务: 每日自动拉取最新数据
    - 断点续传: 失败自动重试

    用法:
        updater = DataUpdater()
        updater.set_watchlist(['000001', '600519', '000858'])
        updater.update_all()  # 增量更新所有关注股票
        updater.run_daily()   # 启动每日自动更新
    """

    def __init__(self, storage: Optional[DataStorage] = None):
        self.storage = storage or DataStorage()
        self.source = AkshareSource()
        self.watchlist_path = Path(config.data.root_dir) / "watchlist.json"
        self._watchlist: List[str] = []

    # ─── 关注列表管理 ─────────────────────────────
    def set_watchlist(self, symbols: List[str]) -> None:
        """设置关注的股票列表，持久化保存"""
        self._watchlist = symbols
        self.watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.watchlist_path, "w", encoding="utf-8") as f:
            json.dump({
                "symbols": symbols,
                "updated_at": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"关注列表已保存: {len(symbols)} 只股票 → {self.watchlist_path}")

    def load_watchlist(self) -> List[str]:
        """加载已保存的关注列表"""
        if self._watchlist:
            return self._watchlist
        if self.watchlist_path.exists():
            with open(self.watchlist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._watchlist = data.get("symbols", [])
                logger.info(f"已加载关注列表: {len(self._watchlist)} 只股票")
        return self._watchlist

    def add_to_watchlist(self, symbols: List[str]) -> None:
        """添加股票到关注列表"""
        existing = set(self.load_watchlist())
        existing.update(symbols)
        self.set_watchlist(list(existing))

    # ─── 增量更新 ────────────────────────────────
    def update_all(
        self,
        symbols: Optional[List[str]] = None,
        years_back: int = 2,
    ) -> dict:
        """增量更新所有股票数据

        逻辑: 检查每只股票的本地最新日期，只下载缺失部分

        Args:
            symbols: 要更新的股票 (默认=关注列表)
            years_back: 如果本地无数据，回溯下载几年

        Returns:
            {"success": int, "failed": int, "new_rows": int}
        """
        symbols = symbols or self.load_watchlist()
        if not symbols:
            logger.error("无关注股票！请先设置: updater.set_watchlist([...])")
            return {"success": 0, "failed": 0, "new_rows": 0}

        today = date.today()
        stats = {"success": 0, "failed": 0, "new_rows": 0}

        logger.info(f"开始增量更新 {len(symbols)} 只股票...")

        for i, symbol in enumerate(symbols):
            # 确定需要下载的日期范围
            latest = self.storage.get_latest_date(symbol)
            if latest:
                start = latest + timedelta(days=1)
                if start >= today:
                    logger.debug(f"{symbol}: 已是最新 ({latest})")
                    continue
            else:
                start = today - timedelta(days=365 * years_back)

            # 下载
            try:
                df = self.source.get_daily([symbol], start, today)
                if not df.empty:
                    count = self.storage.save_daily(df)
                    stats["new_rows"] += len(df)
                    stats["success"] += 1
                    logger.info(
                        f"  [{i+1}/{len(symbols)}] {symbol}: "
                        f"+{len(df)} 条 ({start} ~ {today})"
                    )
                else:
                    logger.debug(f"  {symbol}: 无新数据")
            except Exception as e:
                stats["failed"] += 1
                logger.warning(f"  {symbol}: 更新失败 ({e})")
                continue

            # 股票间延迟
            if i < len(symbols) - 1:
                time.sleep(random.uniform(1.0, 2.0))

        logger.info(
            f"更新完成: {stats['success']} 成功, "
            f"{stats['failed']} 失败, "
            f"{stats['new_rows']} 条新数据"
        )
        return stats

    def update_recent(self, symbols: Optional[List[str]] = None) -> dict:
        """快速更新最近数据 (仅最新30天，适合每日更新)"""
        symbols = symbols or self.load_watchlist()
        if not symbols:
            logger.error("无关注股票！")
            return {"success": 0, "failed": 0, "new_rows": 0}

        today = date.today()
        start = today - timedelta(days=30)

        logger.info(f"快速更新最近30天: {len(symbols)} 只股票")

        success, failed, total_rows = 0, 0, 0
        for symbol in symbols:
            try:
                df = self.source.get_daily([symbol], start, today)
                if not df.empty:
                    self.storage.save_daily(df)
                    total_rows += len(df)
                    success += 1
            except Exception:
                failed += 1

        return {"success": success, "failed": failed, "new_rows": total_rows}

    # ─── 定时任务 ────────────────────────────────
    def run_daily(self, symbols: Optional[List[str]] = None):
        """每日自动更新循环 (阻塞运行)

        每天收盘后(默认15:30)自动拉取最新数据
        按 Ctrl+C 停止
        """
        symbols = symbols or self.load_watchlist()
        if not symbols:
            logger.error("无关注股票！")
            return

        logger.info(f"每日自动更新已启动 | {len(symbols)} 只股票")
        logger.info("按 Ctrl+C 停止")

        try:
            while True:
                now = datetime.now()
                # 每天 15:30 后更新 (A股收盘时间)
                target_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

                if now < target_time:
                    wait_seconds = (target_time - now).total_seconds()
                    logger.info(
                        f"等待 {wait_seconds/60:.0f} 分钟后自动更新 "
                        f"(目标时间: 15:30)"
                    )
                    time.sleep(wait_seconds)

                logger.info(f"开始每日数据更新... ({date.today()})")
                result = self.update_recent(symbols)
                logger.info(f"今日更新: {result}")

                # 等24小时后再来
                time.sleep(86400)

        except KeyboardInterrupt:
            logger.info("每日更新已停止")

    # ─── 状态查看 ────────────────────────────────
    def status(self) -> pd.DataFrame:
        """查看所有关注股票的数据覆盖状态"""
        symbols = self.load_watchlist()
        rows = []
        for sym in symbols:
            latest = self.storage.get_latest_date(sym)
            coverage = self.storage.get_coverage(sym)
            rows.append({
                "symbol": sym,
                "latest_date": latest or "无数据",
                "days_ago": (date.today() - latest).days if latest else "-",
                "total_files": coverage.get("count", 0),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("days_ago", ascending=False)
        return df
