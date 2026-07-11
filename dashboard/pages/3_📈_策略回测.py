"""
策略回测页面 — 参数调优、回测运行、绩效展示
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from datetime import date, timedelta

st.set_page_config(page_title="策略回测", page_icon="📈", layout="wide")

st.title("📈 策略回测")

# ─── 侧边栏参数 ───────────────────────
with st.sidebar:
    st.subheader("⚙️ 回测参数")

    strategy = st.selectbox("选择策略", [
        "双均线交叉",
        "布林带均值回归",
        "多因子动量轮动",
    ])

    initial_capital = st.number_input("初始资金(万)", value=100, step=10) * 10000
    commission = st.number_input("佣金费率(万分之)", value=2.5, step=0.5) / 10000

    st.markdown("---")
    st.subheader("📊 策略参数")

    if strategy == "双均线交叉":
        fast = st.slider("快线周期", 2, 30, 5)
        slow = st.slider("慢线周期", 10, 120, 20)
        extra_params = {"fast": fast, "slow": slow}

    elif strategy == "布林带均值回归":
        period = st.slider("布林带周期", 10, 60, 20)
        std_mult = st.slider("标准差倍数", 1.0, 4.0, 2.0, 0.1)
        stop_loss = st.slider("止损(%)", 1, 20, 5) / 100
        take_profit = st.slider("止盈(%)", 5, 50, 10) / 100
        extra_params = {
            "period": period,
            "std_mult": std_mult,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    else:
        top_n = st.slider("持仓数量", 3, 30, 10)
        rebalance = st.slider("调仓周期(天)", 1, 20, 5)
        mom_period = st.slider("动量周期", 5, 60, 20)
        extra_params = {
            "top_n": top_n,
            "rebalance_days": rebalance,
            "momentum_period": mom_period,
        }

# ─── 主区域 ───────────────────────────
st.subheader("📥 数据与股票选择")

col1, col2, col3 = st.columns(3)
with col1:
    symbols_input = st.text_input(
        "股票代码",
        "000001,000002,600519,000858,600036,601318,000333,600900",
        help="多只用逗号分隔"
    )
with col2:
    start_date = st.date_input("起始日期", date.today() - timedelta(days=365 * 2))
with col3:
    end_date = st.date_input("结束日期", date.today())

# ─── 运行回测 ─────────────────────────
if st.button("🚀 运行回测", type="primary", use_container_width=True):
    if not symbols_input:
        st.warning("请输入股票代码")
    else:
        symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

        with st.spinner(f"正在下载数据并运行回测... ({strategy})"):
            try:
                # 下载数据
                from quant.data.sources import AkshareSource

                source = AkshareSource()
                df = source.get_daily(symbols, start_date, end_date)

                if df.empty:
                    st.error("未获取到数据")
                else:
                    st.info(f"获取了 {len(df)} 条数据, {df['symbol'].nunique()} 只股票, {df['date'].nunique()} 个交易日")

                    # 创建回测引擎
                    from quant.backtest import BacktestEngine, PerformanceAnalytics

                    engine = BacktestEngine(
                        initial_cash=initial_capital,
                        commission_rate=commission,
                    )
                    engine.set_data(df)

                    # 设置策略
                    from quant.strategy.examples import (
                        MACrossoverStrategy,
                        MeanReversionStrategy,
                        AlphaMomentumStrategy,
                    )

                    strategy_map = {
                        "双均线交叉": MACrossoverStrategy,
                        "布林带均值回归": MeanReversionStrategy,
                        "多因子动量轮动": AlphaMomentumStrategy,
                    }

                    strategy_cls = strategy_map[strategy]
                    strat = strategy_cls(**extra_params)
                    engine.set_strategy(strat)

                    # 运行
                    result = engine.run(progress=True)

                    # ─── 展示结果 ─────────────
                    st.markdown("---")
                    st.subheader("📊 回测结果")

                    report = result.describe()

                    # 关键指标
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ret_val = report.get("total_return", "N/A")
                        st.metric("总收益率", ret_val)
                    with col2:
                        st.metric("夏普比率", report.get("sharpe_ratio", "N/A"))
                    with col3:
                        st.metric("最大回撤", report.get("max_drawdown", "N/A"))
                    with col4:
                        st.metric("交易次数", report.get("trade_count", 0))

                    # 净值曲线
                    from quant.visualization.charts import ChartBuilder

                    if not result.equity_curve.empty:
                        st.subheader("📈 净值 & 回撤")

                        # 净值
                        fig1 = ChartBuilder.equity_curve(result.equity_curve)
                        st.plotly_chart(fig1, use_container_width=True)

                        # 回撤
                        fig2 = ChartBuilder.drawdown(result.equity_curve)
                        st.plotly_chart(fig2, use_container_width=True)

                        # 月度热力图
                        st.subheader("🗓 月度收益热力图")
                        fig3 = ChartBuilder.monthly_heatmap(result.equity_curve)
                        if fig3.data:
                            st.plotly_chart(fig3, use_container_width=True)

                        # 绩效表
                        with st.expander("📋 完整绩效指标"):
                            st.json(report)

                    # 交易记录
                    if not result.trade_log.empty:
                        with st.expander("📝 交易记录"):
                            st.dataframe(result.trade_log, use_container_width=True)

            except ImportError as e:
                st.error(f"缺少依赖: {e}\n\n请运行: pip install -r requirements.txt")
            except Exception as e:
                st.error(f"回测失败: {e}")
                import traceback

                st.code(traceback.format_exc())
