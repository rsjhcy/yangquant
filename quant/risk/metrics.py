"""
风险度量
VaR / CVaR / 最大回撤 / 波动率 / 贝塔 / 相关性等
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class RiskMetrics:
    """风险度量工具集

    支持:
    - VaR (历史模拟 / 参数法 / 蒙特卡洛)
    - CVaR (Expected Shortfall)
    - 最大回撤 & 回撤持续期
    - 贝塔 & 阿尔法
    - 波动率 (历史 / 指数加权)
    - 相关性 & 协方差矩阵
    """

    @staticmethod
    def var_historical(
        returns: pd.Series,
        confidence: float = 0.95,
        holding_period: int = 1,
    ) -> float:
        """VaR — 历史模拟法

        Args:
            returns: 日收益率序列
            confidence: 置信水平 (默认95%)
            holding_period: 持有天数

        Returns:
            VaR值 (负值)
        """
        if len(returns) < 2:
            return 0.0
        var_daily = np.percentile(returns, 100 * (1 - confidence))
        return var_daily * np.sqrt(holding_period)

    @staticmethod
    def var_parametric(
        returns: pd.Series,
        confidence: float = 0.95,
        distribution: str = "normal",
    ) -> float:
        """VaR — 参数法 (方差-协方差)

        Args:
            distribution: 'normal' | 't'
        """
        if len(returns) < 2:
            return 0.0

        mu = returns.mean()
        sigma = returns.std()

        if distribution == "normal":
            z_score = stats.norm.ppf(1 - confidence)
        else:
            # t分布 (假设df=5)
            df_t = 5
            z_score = stats.t.ppf(1 - confidence, df_t)

        return mu - z_score * sigma

    @staticmethod
    def cvar_historical(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """CVaR (Expected Shortfall) — 超过VaR的平均损失"""
        var = RiskMetrics.var_historical(returns, confidence)
        tail = returns[returns <= var]
        if len(tail) == 0:
            return var
        return tail.mean()

    @staticmethod
    def max_drawdown(values: pd.Series) -> Tuple[float, int, int]:
        """最大回撤

        Returns:
            (max_drawdown, peak_idx, trough_idx)
        """
        arr = values.values
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / peak
        max_dd_idx = dd.argmin()
        max_dd = dd[max_dd_idx]

        # 找峰值索引
        peak_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if dd[i] == 0:
                peak_idx = i
                break

        return float(max_dd), peak_idx, max_dd_idx

    @staticmethod
    def drawdown_series(values: pd.Series) -> pd.Series:
        """回撤序列"""
        arr = values.values
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / peak
        return pd.Series(dd, index=values.index, name="drawdown")

    @staticmethod
    def beta_alpha(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02,
    ) -> Dict[str, float]:
        """计算 Beta 和 Alpha

        Returns:
            {'beta': ..., 'alpha': ..., 'r_squared': ...}
        """
        aligned = pd.DataFrame({
            "strategy": strategy_returns,
            "benchmark": benchmark_returns,
        }).dropna()

        if len(aligned) < 3:
            return {"beta": 1.0, "alpha": 0.0, "r_squared": 0.0}

        # 回归
        X = aligned["benchmark"].values
        y = aligned["strategy"].values
        X = np.column_stack([np.ones(len(X)), X])

        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        alpha_daily = beta[0]
        beta_value = beta[1]

        # 年化 Alpha
        alpha_annual = alpha_daily * 252

        # R²
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "beta": float(beta_value),
            "alpha_daily": float(alpha_daily),
            "alpha_annual": float(alpha_annual),
            "r_squared": float(r_squared),
        }

    @staticmethod
    def volatility(
        returns: pd.Series,
        method: str = "historical",
        window: int = 252,
    ) -> float:
        """年化波动率

        Args:
            method: 'historical' | 'ewma' (指数加权)
        """
        if method == "ewma":
            # EWMA volatility with lambda=0.94
            ewma_std = returns.ewm(alpha=0.06).std().iloc[-1]
            return float(ewma_std * np.sqrt(252))
        else:
            return float(returns.std() * np.sqrt(252))

    @staticmethod
    def covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """协方差矩阵"""
        return returns_df.cov()

    @staticmethod
    def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """相关性矩阵"""
        return returns_df.corr()
