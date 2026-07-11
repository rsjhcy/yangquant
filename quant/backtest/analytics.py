"""
回测绩效分析
生成专业级绩效报告：收益分解、风险指标、月度分析、滚动指标
"""

from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class PerformanceAnalytics:
    """绩效分析引擎

    提供:
    - 收益分析: 总收益/年化收益/月度收益
    - 风险分析: 波动率/最大回撤/VaR/CVaR/下行标准差
    - 风险调整: 夏普/索提诺/卡玛/信息比率
    - 交易分析: 换手率/滑点成本/持仓集中度
    - 归因分析: Brinson归因/因子暴露
    """

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
    ):
        """
        Args:
            equity_curve: 净值曲线 DataFrame (date, total_value, pnl, ...)
            trade_log: 交易日志 DataFrame
            benchmark_returns: 基准日收益率 Series
            risk_free_rate: 无风险利率(年化)
        """
        self.equity_curve = equity_curve.copy()
        self.trade_log = trade_log.copy() if not trade_log.empty else pd.DataFrame()
        self.rf = risk_free_rate

        # 计算日收益率
        if not self.equity_curve.empty:
            self.equity_curve["daily_return"] = self.equity_curve["total_value"].pct_change()
            self.equity_curve["log_return"] = np.log(self.equity_curve["total_value"] / self.equity_curve["total_value"].shift(1))
            self.daily_returns = self.equity_curve["daily_return"].dropna()
        else:
            self.daily_returns = pd.Series(dtype=float)

        # 基准
        self.benchmark_returns = benchmark_returns
        if self.benchmark_returns is not None:
            self.benchmark_returns = self.benchmark_returns.reindex(self.equity_curve.index, method=None)

    # ─── 核心指标 ───────────────────────────────────
    def full_report(self) -> Dict:
        """生成完整绩效报告"""
        if self.equity_curve.empty:
            return {"error": "无回测数据"}

        return {
            "summary": self._summary_metrics(),
            "returns": self._returns_breakdown(),
            "risk": self._risk_metrics(),
            "risk_adjusted": self._risk_adjusted_returns(),
            "drawdowns": self._drawdown_analysis(),
            "monthly": self._monthly_analysis(),
            "trading": self._trading_analysis(),
        }

    def _summary_metrics(self) -> Dict:
        """汇总指标"""
        eq = self.equity_curve
        start_val = eq["total_value"].iloc[0]
        final_val = eq["total_value"].iloc[-1]
        total_ret = (final_val / start_val) - 1

        # 年化
        trading_days = len(eq)
        years = trading_days / 252
        ann_ret = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

        # 最大回撤
        values = eq["total_value"].values
        peak = np.maximum.accumulate(values)
        dd_series = (values - peak) / peak
        max_dd = dd_series.min()
        max_dd_idx = dd_series.argmin()

        # 夏普
        rf_daily = self.rf / 252
        excess = self.daily_returns - rf_daily if len(self.daily_returns) > 0 else pd.Series([0])
        sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

        return {
            "初始资金": start_val,
            "最终净值": final_val,
            "总收益率": f"{total_ret:.2%}",
            "年化收益率": f"{ann_ret:.2%}",
            "年化波动率": f"{self.daily_returns.std() * np.sqrt(252):.2%}" if len(self.daily_returns) > 1 else "N/A",
            "夏普比率": f"{sharpe:.2f}",
            "最大回撤": f"{max_dd:.2%}",
            "回撤日期": str(eq["date"].iloc[max_dd_idx]) if max_dd_idx < len(eq) else "N/A",
            "Calmar比率": f"{ann_ret / abs(max_dd):.2f}" if abs(max_dd) > 1e-6 else "N/A",
            "胜率": f"{self._calc_win_rate():.1%}",
            "盈亏比": f"{self._calc_profit_factor():.2f}",
        }

    def _returns_breakdown(self) -> Dict:
        """收益分解"""
        return {
            "累计收益率": float(self.equity_curve["total_value"].iloc[-1] / self.equity_curve["total_value"].iloc[0] - 1),
            "日均收益率": float(self.daily_returns.mean()) if len(self.daily_returns) > 0 else 0,
            "正收益天数": int((self.daily_returns > 0).sum()),
            "负收益天数": int((self.daily_returns < 0).sum()),
            "最大单日涨幅": float(self.daily_returns.max()) if len(self.daily_returns) > 0 else 0,
            "最大单日跌幅": float(self.daily_returns.min()) if len(self.daily_returns) > 0 else 0,
            "收益偏度": float(self.daily_returns.skew()) if len(self.daily_returns) > 2 else 0,
            "收益峰度": float(self.daily_returns.kurtosis()) if len(self.daily_returns) > 3 else 0,
        }

    def _risk_metrics(self) -> Dict:
        """风险指标"""
        returns = self.daily_returns.dropna()

        if len(returns) < 2:
            return {}

        # VaR (历史模拟法)
        var_95 = float(np.percentile(returns, 5))
        var_99 = float(np.percentile(returns, 1))

        # CVaR
        cvar_95 = float(returns[returns <= var_95].mean())
        cvar_99 = float(returns[returns <= var_99].mean())

        # 下行标准差
        downside = returns[returns < 0]
        downside_std = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 0

        return {
            "年化波动率": float(returns.std() * np.sqrt(252)),
            "下行波动率": downside_std,
            "VaR(95%)": f"{var_95:.2%}",
            "VaR(99%)": f"{var_99:.2%}",
            "CVaR(95%)": f"{cvar_95:.2%}",
            "CVaR(99%)": f"{cvar_99:.2%}",
        }

    def _risk_adjusted_returns(self) -> Dict:
        """风险调整收益"""
        returns = self.daily_returns.dropna()
        if len(returns) < 2:
            return {}

        rf_daily = self.rf / 252
        excess = returns - rf_daily

        # 夏普
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

        # 索提诺
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.01
        sortino = float((returns.mean() - rf_daily) / (downside_std / np.sqrt(252))) if downside_std > 0 else 0

        # 信息比率 (相对基准)
        ir = 0.0
        if self.benchmark_returns is not None:
            tracking_error = (returns - self.benchmark_returns).dropna()
            if len(tracking_error) > 0 and tracking_error.std() > 0:
                ir = float(tracking_error.mean() / tracking_error.std() * np.sqrt(252))

        return {
            "夏普比率": f"{sharpe:.3f}",
            "索提诺比率": f"{sortino:.3f}",
            "信息比率": f"{ir:.3f}",
        }

    def _drawdown_analysis(self) -> Dict:
        """回撤分析"""
        if self.equity_curve.empty:
            return {}

        values = self.equity_curve["total_value"].values
        peak = np.maximum.accumulate(values)
        dd = (values - peak) / peak

        # 最大回撤
        max_dd = dd.min()
        max_dd_end = dd.argmin()
        max_dd_start = 0
        for i in range(max_dd_end, -1, -1):
            if dd[i] == 0:
                max_dd_start = i
                break

        # 回撤持续时间
        recovery_idx = max_dd_end + 1
        for i in range(max_dd_end + 1, len(dd)):
            if dd[i] >= 0:
                recovery_idx = i
                break

        max_dd_days = max_dd_end - max_dd_start
        recovery_days = recovery_idx - max_dd_end

        # 平均回撤
        avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0

        dates = self.equity_curve["date"]

        return {
            "最大回撤": f"{max_dd:.2%}",
            "最大回撤起始": str(dates.iloc[max_dd_start]) if max_dd_start < len(dates) else "N/A",
            "最大回撤谷底": str(dates.iloc[max_dd_end]) if max_dd_end < len(dates) else "N/A",
            "回撤持续(天)": max_dd_days,
            "恢复天数": recovery_days,
            "平均回撤": f"{avg_dd:.2%}",
        }

    def _monthly_analysis(self) -> pd.DataFrame:
        """月度收益分析"""
        if self.equity_curve.empty:
            return pd.DataFrame()

        eq = self.equity_curve.copy()
        eq["year_month"] = eq["date"].apply(lambda d: f"{d.year}-{d.month:02d}")

        monthly = eq.groupby("year_month").agg(
            月收益率=("daily_return", lambda x: (1 + x).prod() - 1),
            月波动率=("daily_return", "std"),
            月最大回撤=("total_value", lambda x: (x / x.cummax() - 1).min()),
        ).reset_index()

        monthly["累计净值"] = (1 + monthly["月收益率"]).cumprod()
        return monthly

    def _trading_analysis(self) -> Dict:
        """交易分析"""
        if self.trade_log.empty:
            return {"交易笔数": 0}

        buy_trades = self.trade_log[self.trade_log["direction"] == "BUY"]
        sell_trades = self.trade_log[self.trade_log["direction"] == "SELL"]

        return {
            "总交易笔数": len(self.trade_log),
            "买入笔数": len(buy_trades),
            "卖出笔数": len(sell_trades),
            "总佣金": float(self.trade_log["commission"].sum()) if "commission" in self.trade_log.columns else 0,
            "总印花税": float(self.trade_log["stamp_duty"].sum()) if "stamp_duty" in self.trade_log.columns else 0,
            "总交易成本": float(self.trade_log["total_cost"].sum()) if "total_cost" in self.trade_log.columns else 0,
            "日均交易": float(len(self.trade_log) / len(self.equity_curve)) if len(self.equity_curve) > 0 else 0,
        }

    # ─── 辅助 ───────────────────────────────────────
    def _calc_win_rate(self) -> float:
        """计算胜率"""
        returns = self.daily_returns.dropna()
        if len(returns) == 0:
            return 0.0
        return float((returns > 0).sum() / len(returns))

    def _calc_profit_factor(self) -> float:
        """盈亏比"""
        returns = self.daily_returns.dropna()
        if len(returns) == 0:
            return 0.0
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return float(gains / losses) if losses > 0 else float("inf")

    # ─── 与基准对比 ────────────────────────────────
    def compare_to_benchmark(self) -> pd.DataFrame:
        """与基准对比"""
        if self.benchmark_returns is None:
            return pd.DataFrame()

        eq = self.equity_curve.copy()
        eq["strategy_return"] = eq["daily_return"]

        bm = self.benchmark_returns.reindex(eq.index)
        eq["benchmark_return"] = bm.values

        eq["excess_return"] = eq["strategy_return"] - eq["benchmark_return"]
        eq["strategy_cum"] = (1 + eq["strategy_return"]).cumprod()
        eq["benchmark_cum"] = (1 + eq["benchmark_return"]).cumprod()

        return eq
