"""
实盘监控页面 — 持仓、订单、PnL 追踪
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="实盘监控", page_icon="💹", layout="wide")

st.title("💹 实盘监控")

# ─── 状态指示 ─────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    status = st.radio("交易模式", ["📝 模拟交易", "⚡ 实盘交易"], horizontal=True)

with col2:
    st.metric("总资产", "¥1,000,000.00", delta="0.00%")

with col3:
    st.metric("当日PnL", "¥0.00", delta="0.00%")

# ─── 标签页 ───────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 持仓", "📝 订单", "📈 PnL曲线"])

with tab1:
    st.subheader("当前持仓")

    if status == "📝 模拟交易":
        st.info("""
        **模拟交易** 需要在 Python 中启动:
        ```python
        from quant.execution import PaperTradingEngine

        engine = PaperTradingEngine(initial_cash=1_000_000)
        engine.connect()

        # 下单
        engine.submit_order("000001", "BUY", 1000, 10.50)

        # 查看账户
        print(engine.get_account())
        print(engine.get_positions())
        ```
        """)
    else:
        st.warning("⚡ 实盘交易需要配置券商API。\n\n请确保已安装对应SDK并在 config.yaml 中配置。")

    # 模拟持仓表
    st.markdown("---")
    st.subheader("模拟持仓")
    demo_positions = pd.DataFrame({
        "股票代码": ["000001", "600519"],
        "名称": ["平安银行", "贵州茅台"],
        "持仓数量": [10000, 500],
        "成本价": [10.50, 1600.00],
        "现价": [10.80, 1580.00],
        "市值": [108000, 790000],
        "浮动盈亏": [3000, -10000],
    })
    st.dataframe(demo_positions, use_container_width=True)

with tab2:
    st.subheader("订单记录")
    demo_orders = pd.DataFrame({
        "订单ID": ["PAPER-A1B2C3D4", "PAPER-E5F6G7H8"],
        "股票": ["000001", "600519"],
        "方向": ["BUY", "SELL"],
        "数量": [10000, 500],
        "价格": [10.50, 1600.00],
        "状态": ["FILLED", "FILLED"],
        "时间": ["2024-01-15 09:35", "2024-01-16 14:50"],
    })
    st.dataframe(demo_orders, use_container_width=True)

with tab3:
    st.subheader("PnL 追踪")
    st.info("启动模拟交易后，PnL 曲线将在此展示。")
