"""
图表封装 — 基于 Plotly
提供回测报告常用图表
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChartBuilder:
    """Plotly 图表构建器"""

    # ─── 净值曲线 ───────────────────────────────────
    @staticmethod
    def equity_curve(
        equity_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        title: str = "策略净值曲线",
    ) -> go.Figure:
        """净值曲线图"""
        fig = go.Figure()

        # 策略净值
        if "total_value" in equity_df.columns:
            nav = equity_df["total_value"] / equity_df["total_value"].iloc[0]
            fig.add_trace(go.Scatter(
                x=equity_df["date"],
                y=nav,
                mode="lines",
                name="策略净值",
                line=dict(color="#5470c6", width=2),
            ))

        # 基准净值
        if benchmark_df is not None and "total_value" in benchmark_df.columns:
            bm_nav = benchmark_df["total_value"] / benchmark_df["total_value"].iloc[0]
            fig.add_trace(go.Scatter(
                x=benchmark_df["date"],
                y=bm_nav,
                mode="lines",
                name="基准净值",
                line=dict(color="#91cc75", width=1.5, dash="dash"),
            ))

        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="净值",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    # ─── 回撤图 ────────────────────────────────────
    @staticmethod
    def drawdown(
        equity_df: pd.DataFrame,
        title: str = "回撤曲线",
    ) -> go.Figure:
        """回撤面积图"""
        values = equity_df["total_value"].values
        peak = np.maximum.accumulate(values)
        dd = (values - peak) / peak

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df["date"],
            y=dd,
            fill="tozeroy",
            mode="lines",
            name="回撤",
            line=dict(color="#ee6666", width=1),
            fillcolor="rgba(238,102,102,0.3)",
        ))

        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="回撤",
            yaxis_tickformat=".0%",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    # ─── 月度收益热力图 ────────────────────────────
    @staticmethod
    def monthly_heatmap(
        equity_df: pd.DataFrame,
        title: str = "月度收益热力图",
    ) -> go.Figure:
        """月度收益热力图"""
        eq = equity_df.copy()
        eq["date"] = pd.to_datetime(eq["date"])
        eq["year"] = eq["date"].dt.year
        eq["month"] = eq["date"].dt.month

        monthly = eq.groupby(["year", "month"])["total_value"].agg("last")
        monthly_ret = monthly.unstack().pct_change(axis=1)
        # 简化：用月内收益
        monthly_simple = eq.set_index("date")["total_value"].resample("ME").last().pct_change()
        months = monthly_simple.index

        # 构建矩阵
        pivot_data = []
        for i in range(len(months)):
            pivot_data.append({
                "year": months[i].year,
                "month": months[i].month,
                "return": monthly_simple.iloc[i],
            })

        if not pivot_data:
            return go.Figure()

        df = pd.DataFrame(pivot_data)
        pivot = df.pivot(index="year", columns="month", values="return")

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f"{m}月" for m in pivot.columns],
            y=pivot.index,
            colorscale="RdYlGn",
            text=[[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 11},
        ))

        fig.update_layout(
            title=title,
            xaxis_title="月份",
            yaxis_title="年份",
            template="plotly_white",
        )

        return fig

    # ─── 绩效仪表盘 ──────────────────────────────────
    @staticmethod
    def performance_dashboard(
        metrics: dict,
        title: str = "绩效仪表盘",
    ) -> go.Figure:
        """关键指标仪表盘"""
        indicators = []
        for key, value in metrics.items():
            if isinstance(value, str):
                indicators.append((key, value))

        # 用表格展示
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["指标", "数值"],
                fill_color="paleturquoise",
                align="left",
                font=dict(size=13),
            ),
            cells=dict(
                values=[
                    [i[0] for i in indicators],
                    [i[1] for i in indicators],
                ],
                fill_color="lavender",
                align="left",
                font=dict(size=12),
            ),
        )])

        fig.update_layout(title=title)
        return fig

    # ─── 收益分布直方图 ────────────────────────────
    @staticmethod
    def returns_distribution(
        returns: pd.Series,
        title: str = "收益率分布",
    ) -> go.Figure:
        """收益分布直方图 + 正态拟合"""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns.dropna(),
            nbinsx=50,
            name="实际分布",
            histnorm="probability density",
            marker_color="#5470c6",
            opacity=0.7,
        ))

        # 正态拟合
        x_range = np.linspace(returns.min(), returns.max(), 100)
        from scipy import stats
        mu, sigma = returns.mean(), returns.std()
        y_norm = stats.norm.pdf(x_range, mu, sigma)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_norm,
            mode="lines",
            name=f"正态拟合 (μ={mu:.3f}, σ={sigma:.3f})",
            line=dict(color="#ee6666", width=2),
        ))

        fig.update_layout(
            title=title,
            xaxis_title="收益率",
            yaxis_title="概率密度",
            template="plotly_white",
        )
        return fig

    # ─── IC 曲线 ──────────────────────────────────
    @staticmethod
    def ic_curve(
        ic_series: pd.Series,
        title: str = "IC曲线",
    ) -> go.Figure:
        """IC时间序列"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.05)

        # IC 线
        fig.add_trace(
            go.Scatter(x=ic_series.index, y=ic_series.values,
                       mode="lines", name="IC",
                       line=dict(color="#5470c6", width=1)),
            row=1, col=1,
        )

        # 累计IC
        cum_ic = ic_series.cumsum()
        fig.add_trace(
            go.Scatter(x=cum_ic.index, y=cum_ic.values,
                       mode="lines", name="累计IC",
                       line=dict(color="#ee6666", width=1.5)),
            row=2, col=1,
        )

        # 零线
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                       row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                       row=2, col=1)

        fig.update_layout(
            title=title,
            template="plotly_white",
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="IC", row=1, col=1)
        fig.update_yaxes(title_text="累计IC", row=2, col=1)
        return fig
