"""
集合竞价分析器
次日 9:15-9:25 运行: 获取竞价分钟数据 → 分析竞价强度 → 验证/淘汰 → 生成最终推荐
"""

import time
import random
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class AuctionAnalyzer:
    """集合竞价分析器

    分析维度:
    1. 价格趋势: 竞价末期(9:20-9:25)价格是否稳步抬高
    2. 量能: 竞价量比是否放大
    3. 高开幅度: 2%-5% 最佳区间
    4. 稳定性: 9:20后不可撤单阶段，价格是否稳定
    """

    # 竞价关键时间点
    AUCTION_START = "09:15:00"
    AUCTION_LOCK = "09:20:00"     # 不可撤单
    AUCTION_END = "09:25:00"

    def __init__(self):
        self._auction_data: Dict[str, pd.DataFrame] = {}
        self._spot_data: Dict[str, Dict] = {}

    # ─── 获取竞价数据 ──────────────────────────
    def fetch_auction_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取单只股票的集合竞价分钟数据

        Returns:
            DataFrame with columns: 时间, 开盘, 收盘, 最高, 最低, 成交量, 成交额
        """
        import akshare as ak

        try:
            raw = ak.stock_zh_a_hist_pre_min_em(
                symbol=symbol,
                start_time=self.AUCTION_START,
                end_time=self.AUCTION_END,
            )

            if raw is None or raw.empty:
                return None

            # 列名映射 (东方财富返回中文列名)
            col_map = {
                "时间": "time",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "最新价": "last",
            }
            df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

            # 类型转换
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])

            for col in ["open", "close", "high", "low", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            logger.warning(f"{symbol}: 竞价数据获取失败 ({e})")
            return None

    def fetch_all_auctions(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """批量获取竞价数据"""
        logger.info(f"获取 {len(symbols)} 只股票的竞价数据...")
        data = {}
        for i, sym in enumerate(symbols):
            df = self.fetch_auction_data(sym)
            if df is not None:
                data[sym] = df
            if i < len(symbols) - 1:
                time.sleep(random.uniform(0.3, 0.6))
        logger.info(f"成功获取 {len(data)}/{len(symbols)} 只")
        self._auction_data = data
        return data

    # ─── 价格趋势分析 ─────────────────────────
    def analyze_price_trend(self, df: pd.DataFrame) -> Tuple[float, str]:
        """分析竞价末期价格走势

        关注 9:20-9:25（不可撤单阶段）

        Returns:
            (score 0-100, description)
        """
        if df.empty or "close" not in df.columns or "time" not in df.columns:
            return 50, "数据不足"

        # 筛选 9:20 之后的不可撤单阶段
        lock_time = pd.Timestamp(
            df["time"].iloc[0].date().isoformat() + " 09:20:00"
        )
        late = df[df["time"] >= lock_time]

        if len(late) < 3:
            late = df.tail(max(3, len(df) // 2))

        prices = late["close"].dropna().values
        if len(prices) < 2:
            return 50, "竞价数据不足"

        # 线性回归斜率
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]

        # 价格波动率
        pct_changes = np.diff(prices) / prices[:-1]
        volatility = np.std(pct_changes) if len(pct_changes) > 0 else 0

        score = 50
        desc = ""

        # 斜率评分
        if slope > 0.01:
            score += 25
            desc = "竞价稳步抬升"
        elif slope > 0:
            score += 15
            desc = "竞价小幅走高"
        elif slope > -0.01:
            score += 5
            desc = "竞价横盘"
        elif slope > -0.03:
            score -= 10
            desc = "竞价小幅走低"
        else:
            score -= 25
            desc = "竞价持续走低⚠"

        # 波动扣分
        if volatility > 0.005:
            score -= 15
            desc += "，波动较大"
        elif volatility < 0.002:
            score += 10
            desc += "，走势稳定"

        return max(0, min(100, score)), desc

    # ─── 量能分析 ────────────────────────────
    def analyze_volume(self, df: pd.DataFrame) -> Tuple[float, str]:
        """分析竞价量能

        Returns:
            (score 0-100, description)
        """
        if df.empty or "volume" not in df.columns:
            return 50, "无数据"

        volumes = df["volume"].dropna().values
        if len(volumes) < 2:
            return 50, "数据不足"

        # 最后5分钟 vs 前5分钟
        n = len(volumes)
        late_vol = volumes[-min(5, n):].sum()
        early_vol = volumes[:min(5, n)].sum()

        if early_vol > 0:
            vol_ratio = late_vol / early_vol
        else:
            vol_ratio = 1.0

        # 总成交量
        total_vol = volumes.sum()

        score = 50
        desc = ""

        if vol_ratio > 2.0:
            score += 30
            desc = "竞价末期放量明显"
        elif vol_ratio > 1.3:
            score += 20
            desc = "竞价末期量能放大"
        elif vol_ratio > 0.8:
            score += 5
            desc = "竞价量能平稳"
        else:
            score -= 10
            desc = "竞价末期缩量"

        # 绝对量
        if total_vol > 50000:
            score += 10
            desc += "，交投活跃"
        elif total_vol < 10000:
            score -= 10
            desc += "，成交清淡"

        return max(0, min(100, score)), desc

    # ─── 高开幅度分析 ────────────────────────
    def check_open_range(
        self, df: pd.DataFrame, prev_close: float
    ) -> Tuple[float, str]:
        """评估高开幅度

        Returns:
            (score, description)
        """
        if df.empty or "close" not in df.columns or prev_close <= 0:
            return 50, ""

        last_price = df["close"].dropna().iloc[-1]
        open_pct = (last_price - prev_close) / prev_close * 100

        score = 50
        desc = f"高开{open_pct:+.1f}%"

        if 2 <= open_pct <= 5:
            score += 30
            desc += " ✅最佳区间"
        elif 0.5 <= open_pct < 2:
            score += 15
            desc += " 温和高开"
        elif 0 <= open_pct < 0.5:
            score += 5
            desc = f"平开{open_pct:+.1f}%"
        elif 5 < open_pct <= 7:
            score += 0
            desc += " ⚠偏高"
        elif open_pct > 7:
            score -= 20
            desc += " ⚠追高风险大"
        elif -2 <= open_pct < 0:
            score -= 10
            desc += " 低开需谨慎"
        else:
            score -= 25
            desc += " ⚠大幅低开"

        return max(0, min(100, score)), desc

    # ─── 综合验证 ────────────────────────────
    def validate(
        self, candidates: List[Dict], prev_close_map: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """验证昨日候选，生成今日最终推荐

        Args:
            candidates: 收盘筛选出的候选 [{symbol, name, close, ...}]
            prev_close_map: {symbol: prev_close_price}

        Returns:
            验证后的推荐列表 [{symbol, name, ..., auction_score, auction_detail, recommendation}]
        """
        symbols = [c["symbol"] for c in candidates]
        self.fetch_all_auctions(symbols)
        prev_close_map = prev_close_map or {}

        results = []
        for candidate in candidates:
            sym = candidate["symbol"]
            df = self._auction_data.get(sym)

            if df is None or df.empty:
                logger.warning(f"{sym}: 无竞价数据，保留但标记")
                candidate["auction_score"] = 0
                candidate["auction_detail"] = "竞价数据缺失"
                candidate["recommendation"] = "谨慎关注(无竞价数据)"
                candidate["open_pct"] = "-"
                results.append(candidate)
                continue

            prev_close = prev_close_map.get(
                sym,
                float(candidate.get("close", 0) or 0),
            )

            # 三维分析
            trend_score, trend_desc = self.analyze_price_trend(df)
            vol_score, vol_desc = self.analyze_volume(df)
            open_score, open_desc = self.check_open_range(df, prev_close)

            # 综合竞价得分
            auction_score = trend_score * 0.35 + vol_score * 0.30 + open_score * 0.35

            # 淘汰规则
            if open_score < 20:
                recommendation = "❌ 不建议(竞价高开异常)"
            elif auction_score < 35:
                recommendation = "❌ 不建议(竞价信号弱)"
            elif auction_score >= 70:
                recommendation = "✅ 强烈推荐"
            elif auction_score >= 55:
                recommendation = "👍 可以关注"
            else:
                recommendation = "🤔 谨慎关注"

            candidate["auction_score"] = round(auction_score, 1)
            candidate["auction_detail"] = f"{trend_desc} | {vol_desc} | {open_desc}"
            candidate["recommendation"] = recommendation
            candidate["open_pct"] = open_desc

            logger.info(
                f"{sym}: 竞价{auction_score:.0f}分 "
                f"(趋势{trend_score:.0f} 量能{vol_score:.0f} 高开{open_score:.0f}) → {recommendation}"
            )

            results.append(candidate)

        # 按竞价得分排序
        results.sort(key=lambda x: x.get("auction_score", 0), reverse=True)
        return results

    # ─── 获取前收盘价 ────────────────────────
    def get_prev_close_map(self, symbols: List[str]) -> Dict[str, float]:
        """获取一组股票的最新收盘价"""
        import akshare as ak

        price_map = {}
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    code = str(row.get("代码", ""))
                    if code in symbols:
                        price_map[code] = float(row.get("最新价", 0))
        except Exception as e:
            logger.warning(f"获取实时报价失败: {e}")

        return price_map
