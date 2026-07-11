"""
信号生成框架
提供常用信号处理: 去噪/平滑/交叉检测/阈值过滤
"""

from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class SignalGenerator:
    """信号生成器 — 提供常用信号处理工具

    可用于:
    - 平滑信号 (SMA/EMA去噪)
    - 交叉检测 (金叉/死叉)
    - 阈值过滤 (突破/告警)
    - 信号合并 (多信号投票)
    - 信号延迟 (避免未来信息)
    """

    # ─── 均线/平滑 ─────────────────────────────────
    @staticmethod
    def sma(data: List[float], period: int) -> np.ndarray:
        """简单移动平均（边界值用有效窗口计算，不做零填充）"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        arr = np.array(data, dtype=float)
        result = np.full(len(arr), np.nan)
        # 只计算有效窗口，不做零填充
        for i in range(period - 1, len(arr)):
            result[i] = np.mean(arr[i - period + 1 : i + 1])
        return result

    @staticmethod
    def ema(data: List[float], period: int) -> np.ndarray:
        """指数移动平均"""
        if len(data) < 2:
            return np.array(data, dtype=float)
        arr = np.array(data, dtype=float)
        alpha = 2.0 / (period + 1)
        result = np.full(len(arr), np.nan)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            if np.isnan(result[i - 1]):
                result[i] = arr[i]
            else:
                result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    # ─── 交叉检测 ──────────────────────────────────
    @staticmethod
    def detect_cross(
        fast: List[float],
        slow: List[float],
    ) -> Tuple[Optional[str], int]:
        """检测最新交叉信号

        Args:
            fast: 快线序列
            slow: 慢线序列

        Returns:
            ('golden_cross' | 'death_cross' | None, index)
            金叉 = 快线上穿慢线, 死叉 = 快线下穿慢线
        """
        if len(fast) < 2 or len(slow) < 2:
            return (None, -1)

        i = len(fast) - 1
        prev_fast, curr_fast = fast[i - 1], fast[i]
        prev_slow, curr_slow = slow[i - 1], slow[i]

        if pd.isna(prev_fast) or pd.isna(curr_fast):
            return (None, -1)
        if pd.isna(prev_slow) or pd.isna(curr_slow):
            return (None, -1)

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return ("golden_cross", i)      # 金叉 → 买入信号
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            return ("death_cross", i)       # 死叉 → 卖出信号

        return (None, -1)

    # ─── 突破检测 ──────────────────────────────────
    @staticmethod
    def detect_breakout(
        prices: List[float],
        level: float,
        direction: str = "up",
        n_bars: int = 1,
    ) -> bool:
        """检测突破

        Args:
            prices: 价格序列
            level: 突破水平
            direction: 'up' 向上突破 / 'down' 向下突破
            n_bars: 需要连续N根K线确认

        Returns:
            True=突破确认
        """
        if len(prices) < n_bars:
            return False

        recent = prices[-n_bars:]
        if direction == "up":
            return all(p > level for p in recent)
        else:
            return all(p < level for p in recent)

    # ─── 极值检测 ──────────────────────────────────
    @staticmethod
    def detect_extreme(
        values: List[float],
        threshold: float,
        n_bars: int = 5,
    ) -> Optional[str]:
        """检测极端值区域

        Returns:
            'overbought' | 'oversold' | None
        """
        if len(values) < n_bars:
            return None

        recent_avg = np.mean(values[-n_bars:])
        if recent_avg > threshold:
            return "overbought"
        elif recent_avg < -threshold:
            return "oversold"
        return None

    # ─── 信号过滤 ──────────────────────────────────
    @staticmethod
    def filter_signals(
        signals: pd.Series,
        min_interval: int = 3,
    ) -> pd.Series:
        """过滤过于频繁的信号 (最小间隔=min_interval根K线)

        Args:
            signals: 信号序列 (1=buy, -1=sell, 0=hold)
            min_interval: 两次信号间的最小K线数

        Returns:
            过滤后的信号序列
        """
        filtered = signals.copy()
        last_signal_idx = -min_interval - 1

        for i in range(len(signals)):
            if filtered.iloc[i] != 0:
                if i - last_signal_idx < min_interval:
                    filtered.iloc[i] = 0
                else:
                    last_signal_idx = i

        return filtered

    # ─── 多信号投票 ────────────────────────────────
    @staticmethod
    def vote_signals(
        signal_list: List[pd.Series],
        weights: Optional[List[float]] = None,
    ) -> pd.Series:
        """多信号加权投票

        Args:
            signal_list: 多个信号Series
            weights: 投票权重

        Returns:
            综合投票结果
        """
        if not signal_list:
            return pd.Series(dtype=float)

        if weights is None:
            weights = [1.0] * len(signal_list)

        result = pd.DataFrame({
            f"sig_{i}": s.fillna(0)
            for i, s in enumerate(signal_list)
        })

        # 加权求和
        weighted = sum(
            result[f"sig_{i}"] * w
            for i, w in enumerate(weights)
        )
        return weighted / sum(weights)
