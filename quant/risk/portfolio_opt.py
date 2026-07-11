"""
组合优化
均值-方差优化 / 风险平价 / 最大夏普
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class PortfolioOptimizer:
    """组合优化器

    支持:
    - 均值-方差优化 (Markowitz)
    - 风险平价 (Risk Parity)
    - 最大夏普比率
    - 最小方差
    - 约束: 权重上下限/总和=1/标的数量上限
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
    ):
        """
        Args:
            returns: 日收益率 DataFrame (columns=symbols)
            risk_free_rate: 年化无风险利率
        """
        self.returns = returns.dropna()
        self.rf = risk_free_rate
        self.rf_daily = risk_free_rate / 252
        self.mu = self.returns.mean()
        self.sigma = self.returns.cov()
        self.n_assets = len(self.returns.columns)

    # ─── 均值-方差优化 ──────────────────────────────
    def mean_variance(
        self,
        target_return: Optional[float] = None,
        bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[np.ndarray, float, float]:
        """均值-方差优化

        Args:
            target_return: 目标年化收益率 (None=最小方差)
            bounds: 权重范围 (lower, upper)

        Returns:
            (weights, portfolio_return, portfolio_volatility)
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.error("需要 scipy")

        n = self.n_assets
        init_guess = np.ones(n) / n

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds_list = [bounds] * n

        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda w: (self.mu @ w * 252 - target_return),
            })

        def portfolio_vol(w):
            return np.sqrt(w @ self.sigma @ w) * np.sqrt(252)

        result = minimize(
            portfolio_vol,
            init_guess,
            method="SLSQP",
            bounds=bounds_list,
            constraints=constraints,
        )

        w = result.x
        w[w < 1e-6] = 0  # 极小权重归零
        w = w / w.sum()

        port_return = self.mu @ w * 252
        port_vol = portfolio_vol(w)

        return w, port_return, port_vol

    def max_sharpe(self) -> Tuple[np.ndarray, float, float, float]:
        """最大化夏普比率"""
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.error("需要 scipy")

        n = self.n_assets
        init_guess = np.ones(n) / n

        def neg_sharpe(w):
            port_ret = self.mu @ w * 252
            port_vol = np.sqrt(w @ self.sigma @ w) * np.sqrt(252)
            return -(port_ret - self.rf) / port_vol if port_vol > 0 else 0

        result = minimize(
            neg_sharpe,
            init_guess,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        )

        w = result.x
        w[w < 1e-6] = 0
        w = w / w.sum()

        port_return = self.mu @ w * 252
        port_vol = np.sqrt(w @ self.sigma @ w) * np.sqrt(252)
        sharpe = (port_return - self.rf) / port_vol

        return w, port_return, port_vol, sharpe

    def min_variance(self) -> Tuple[np.ndarray, float, float]:
        """最小方差组合"""
        return self.mean_variance(target_return=None)

    # ─── 风险平价 ──────────────────────────────────
    def risk_parity(self) -> Tuple[np.ndarray, dict]:
        """风险平价 — 每类资产贡献相等风险

        Returns:
            (weights, risk_contributions: {symbol: risk_pct})
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.error("需要 scipy")

        n = self.n_assets

        def risk_budget_objective(w):
            w = np.maximum(w, 0)
            w = w / w.sum()
            port_vol = np.sqrt(w @ self.sigma @ w)
            # 边际风险贡献
            mrc = self.sigma @ w / port_vol
            # 风险贡献
            rc = w * mrc
            # 目标: 均等风险贡献
            target_rc = port_vol / n
            return np.sum((rc - target_rc) ** 2)

        init_guess = np.ones(n) / n
        result = minimize(
            risk_budget_objective,
            init_guess,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        )

        w = result.x
        w[w < 1e-6] = 0
        w = w / w.sum()

        # 计算风险贡献
        port_vol = np.sqrt(w @ self.sigma @ w)
        mrc = self.sigma @ w / port_vol
        rc = w * mrc
        rc_pct = rc / rc.sum()

        risk_contrib = {
            self.returns.columns[i]: rc_pct[i]
            for i in range(n)
        }

        return w, risk_contrib

    # ─── 效用函数 ──────────────────────────────────
    def efficient_frontier(
        self,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """生成有效前沿"""
        max_ret = self.mu.max() * 252 * 0.8
        min_ret = 0.0
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []
        for tr in target_returns:
            try:
                w, ret, vol = self.mean_variance(target_return=tr)
                frontier.append({
                    "return": ret,
                    "volatility": vol,
                    "sharpe": (ret - self.rf) / vol if vol > 0 else 0,
                    **{self.returns.columns[i]: w[i] for i in range(self.n_assets)},
                })
            except Exception:
                continue

        return pd.DataFrame(frontier)
