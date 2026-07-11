"""
技术类因子
动量因子 / 波动因子 / 量价因子 / 趋势因子
"""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant.factors.base import (
    BaseFactor,
    FactorCategory,
    FactorDirection,
    FactorResult,
)


# ═══════════════════════════════════════════════════
# 动量因子
# ═══════════════════════════════════════════════════

class MomentumFactor(BaseFactor):
    """动量因子 — N日收益率"""

    name = "momentum"
    category = FactorCategory.MOMENTUM
    direction = FactorDirection.POSITIVE
    requires = ["close"]

    def __init__(self, period: int = 20, skip_recent: int = 1):
        """
        Args:
            period: 回溯周期 (默认20个交易日)
            skip_recent: 跳过最近N日 (避免反转效应)
        """
        self.period = period
        self.skip_recent = skip_recent
        self.name = f"momentum_{period}d"

    def compute(self, data: pd.DataFrame) -> FactorResult:
        self.validate_input(data)

        # 按symbol分组计算
        results = {}
        for symbol, group in data.groupby("symbol"):
            group = group.sort_values("date")
            closes = group["close"].values
            if len(closes) < self.period + self.skip_recent:
                results[symbol] = np.nan
            else:
                # 收益率 = (当前 - N日前) / N日前
                past_price = closes[-(self.period + self.skip_recent)]
                current_price = closes[-self.skip_recent] if self.skip_recent > 0 else closes[-1]
                if past_price > 0:
                    results[symbol] = current_price / past_price - 1
                else:
                    results[symbol] = np.nan

        return FactorResult(
            name=self.name,
            values=pd.Series(results, name="factor"),
            date=data["date"].iloc[-1] if "date" in data.columns else date.today(),
            category=self.category,
            direction=self.direction,
        )


class RSIFactor(BaseFactor):
    """RSI 因子"""

    name = "rsi"
    category = FactorCategory.MOMENTUM
    direction = FactorDirection.NEGATIVE  # 高RSI=超买, 负向
    requires = ["close"]

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"rsi_{period}"

    def compute(self, data: pd.DataFrame) -> FactorResult:
        self.validate_input(data)
        results = {}
        for symbol, group in data.groupby("symbol"):
            group = group.sort_values("date")
            closes = group["close"]
            if len(closes) < self.period + 1:
                results[symbol] = np.nan
                continue

            delta = closes.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)

            avg_gain = gain.rolling(self.period).mean().iloc[-1]
            avg_loss = loss.rolling(self.period).mean().iloc[-1]

            if avg_loss == 0:
                results[symbol] = 100.0
            else:
                rs = avg_gain / avg_loss
                results[symbol] = 100.0 - (100.0 / (1.0 + rs))

        return FactorResult(
            name=self.name,
            values=pd.Series(results, name="factor"),
            date=data["date"].iloc[-1] if "date" in data.columns else date.today(),
            category=self.category,
            direction=self.direction,
        )


# ═══════════════════════════════════════════════════
# 波动因子
# ═══════════════════════════════════════════════════

class VolatilityFactor(BaseFactor):
    """历史波动率因子"""

    name = "volatility"
    category = FactorCategory.VOLATILITY
    direction = FactorDirection.NEGATIVE  # 高波动=高风险
    requires = ["close"]

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"volatility_{period}d"

    def compute(self, data: pd.DataFrame) -> FactorResult:
        self.validate_input(data)
        results = {}
        for symbol, group in data.groupby("symbol"):
            group = group.sort_values("date")
            returns = group["close"].pct_change().dropna()
            if len(returns) < self.period:
                results[symbol] = np.nan
            else:
                results[symbol] = returns.iloc[-self.period:].std() * np.sqrt(252)

        return FactorResult(
            name=self.name,
            values=pd.Series(results, name="factor"),
            date=data["date"].iloc[-1] if "date" in data.columns else date.today(),
            category=self.category,
            direction=self.direction,
        )


class ATRFactor(BaseFactor):
    """ATR (Average True Range) 因子"""

    name = "atr"
    category = FactorCategory.VOLATILITY
    direction = FactorDirection.NEGATIVE
    requires = ["high", "low", "close"]

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"atr_{period}"

    def compute(self, data: pd.DataFrame) -> FactorResult:
        self.validate_input(data)
        results = {}
        for symbol, group in data.groupby("symbol"):
            group = group.sort_values("date")
            high, low, close = group["high"], group["low"], group["close"]

            if len(close) < self.period + 1:
                results[symbol] = np.nan
                continue

            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)

            results[symbol] = tr.rolling(self.period).mean().iloc[-1]

        return FactorResult(
            name=self.name,
            values=pd.Series(results, name="factor"),
            date=data["date"].iloc[-1] if "date" in data.columns else date.today(),
            category=self.category,
            direction=self.direction,
        )


# ═══════════════════════════════════════════════════
# 量价因子
# ═══════════════════════════════════════════════════

class TurnoverFactor(BaseFactor):
    """换手率异常因子 — 高换手有短期动量"""

    name = "turnover_anomaly"
    category = FactorCategory.VOLUME_PRICE
    direction = FactorDirection.POSITIVE
    requires = ["turnover", "close"]

    def __init__(self, short: int = 5, long: int = 20):
        self.short = short
        self.long = long

    def compute(self, data: pd.DataFrame) -> FactorResult:
        self.validate_input(data)
        results = {}
        for symbol, group in data.groupby("symbol"):
            group = group.sort_values("date")
            t = group["turnover"]
            if len(t) < self.long:
                results[symbol] = np.nan
            else:
                # 短期换手相对长期均值的偏离
                short_avg = t.iloc[-self.short:].mean()
                long_avg = t.iloc[-self.long:].mean()
                if long_avg > 0:
                    results[symbol] = short_avg / long_avg - 1
                else:
                    results[symbol] = 0

        return FactorResult(
            name=self.name,
            values=pd.Series(results, name="factor"),
            date=data["date"].iloc[-1] if "date" in data.columns else date.today(),
            category=self.category,
            direction=self.direction,
        )


class VolumePriceFactor(BaseFactor):
    """量价背离因子 — 价格上涨但成交量缩小 → 负向信号"""

    name = "volume_price_divergence"
    category = FactorCategory.VOLUME_PRICE
    direction = FactorDirection.NEGATIVE
    requires = ["close", "volume"]

    def __init__(self, period: int = 10):
        self.period = period

    def compute(self, data: pd.DataFrame) -> FactorResult:
        self.validate_input(data)
        results = {}
        for symbol, group in data.groupby("symbol"):
            group = group.sort_values("date")
            if len(group) < self.period:
                results[symbol] = np.nan
                continue

            recent = group.iloc[-self.period:]
            price_change = recent["close"].pct_change().dropna().mean()
            volume_change = recent["volume"].pct_change().dropna().mean()

            # 价格涨+量缩 = 背离 → 大正值意味着风险
            results[symbol] = price_change - volume_change

        return FactorResult(
            name=self.name,
            values=pd.Series(results, name="factor"),
            date=data["date"].iloc[-1] if "date" in data.columns else date.today(),
            category=self.category,
            direction=self.direction,
        )


# ═══════════════════════════════════════════════════
# 趋势因子
# ═══════════════════════════════════════════════════

class MADeviationFactor(BaseFactor):
    """均线偏离因子 — 价格偏离均线的程度"""

    name = "ma_deviation"
    category = FactorCategory.TREND
    direction = FactorDirection.POSITIVE  # 短期: 正向偏离=趋势延续
    requires = ["close"]

    def __init__(self, ma_period: int = 20):
        self.ma_period = ma_period
        self.name = f"ma_dev_{ma_period}"

    def compute(self, data: pd.DataFrame) -> FactorResult:
        self.validate_input(data)
        results = {}
        for symbol, group in data.groupby("symbol"):
            group = group.sort_values("date")
            closes = group["close"]
            if len(closes) < self.ma_period:
                results[symbol] = np.nan
            else:
                ma = closes.iloc[-self.ma_period:].mean()
                current = closes.iloc[-1]
                if ma > 0:
                    results[symbol] = current / ma - 1
                else:
                    results[symbol] = np.nan

        return FactorResult(
            name=self.name,
            values=pd.Series(results, name="factor"),
            date=data["date"].iloc[-1] if "date" in data.columns else date.today(),
            category=self.category,
            direction=self.direction,
        )


class ADXFactor(BaseFactor):
    """ADX 趋势强度因子"""

    name = "adx"
    category = FactorCategory.TREND
    direction = FactorDirection.POSITIVE
    requires = ["high", "low", "close"]

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"adx_{period}"

    def compute(self, data: pd.DataFrame) -> FactorResult:
        self.validate_input(data)
        results = {}
        for symbol, group in data.groupby("symbol"):
            group = group.sort_values("date")
            if len(group) < self.period * 2:
                results[symbol] = np.nan
                continue

            high, low, close = group["high"], group["low"], group["close"]
            prev_close = close.shift(1)

            # True Range
            tr = pd.DataFrame({
                "a": high - low,
                "b": abs(high - prev_close),
                "c": abs(low - prev_close),
            }).max(axis=1)

            # Directional Movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low

            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            atr = tr.rolling(self.period).mean()
            plus_di = 100 * pd.Series(plus_dm).rolling(self.period).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(self.period).mean() / atr

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
            results[symbol] = dx.iloc[-1]

        return FactorResult(
            name=self.name,
            values=pd.Series(results, name="factor"),
            date=data["date"].iloc[-1] if "date" in data.columns else date.today(),
            category=self.category,
            direction=self.direction,
        )
