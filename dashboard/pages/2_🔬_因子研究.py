"""
因子研究页面 — IC分析、分层回测、因子相关性
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

st.set_page_config(page_title="因子研究", page_icon="🔬", layout="wide")

st.title("🔬 因子研究")

# ─── 数据加载 ─────────────────────────
with st.expander("📥 数据加载", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        symbols_input = st.text_input("股票代码", "000001,000002,600519,000858,600036")
    with col2:
        start_date = st.date_input("起始日期", date.today() - timedelta(days=365))
    with col3:
        end_date = st.date_input("结束日期", date.today())

    if st.button("加载数据并计算因子", type="primary"):
        st.session_state["factor_data_loaded"] = True

        with st.spinner("加载数据..."):
            try:
                from quant.data.sources import AkshareSource

                symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
                source = AkshareSource()
                df = source.get_daily(symbols, start_date, end_date)
                st.session_state["raw_data"] = df
                st.success(f"加载了 {len(df)} 条数据")
            except Exception as e:
                st.error(f"加载失败: {e}")

# ─── 因子计算 ─────────────────────────
if st.session_state.get("factor_data_loaded") and "raw_data" in st.session_state:
    st.markdown("---")
    st.subheader("🔢 计算因子")

    factor_options = st.multiselect(
        "选择因子",
        ["momentum_20d", "rsi_14", "volatility_20d", "ma_deviation_20", "turnover_anomaly"],
        default=["momentum_20d", "volatility_20d"],
    )

    if st.button("计算选中的因子") and factor_options:
        with st.spinner("计算因子中..."):
            try:
                from quant.factors.technical import (
                    MomentumFactor,
                    RSIFactor,
                    VolatilityFactor,
                    MADeviationFactor,
                    TurnoverFactor,
                )

                df = st.session_state["raw_data"]
                factor_map = {
                    "momentum_20d": MomentumFactor(period=20),
                    "rsi_14": RSIFactor(period=14),
                    "volatility_20d": VolatilityFactor(period=20),
                    "ma_deviation_20": MADeviationFactor(ma_period=20),
                    "turnover_anomaly": TurnoverFactor(short=5, long=20),
                }

                results = {}
                for name in factor_options:
                    factor = factor_map[name]
                    result = factor.compute(df)
                    results[name] = result.values

                factor_df = pd.DataFrame(results)
                st.session_state["factor_df"] = factor_df

                st.subheader("因子值预览")
                st.dataframe(factor_df.describe(), use_container_width=True)

                # 相关性热力图
                if len(factor_options) >= 2:
                    import plotly.express as px
                    corr = factor_df.corr()
                    fig = px.imshow(
                        corr,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title="因子相关性热力图",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"因子计算失败: {e}")
                import traceback
                st.code(traceback.format_exc())

# ─── IC分析(简化版) ──────────────────
st.markdown("---")
st.subheader("📈 因子分析 (简化版)")

st.info("""
**IC分析** 需要将因子值和未来收益对齐后计算相关系数。\
完整版请使用 Python API:
```python
from quant.factors.analysis import FactorAnalyzer
analyzer = FactorAnalyzer(factor_values, future_returns)
ic = analyzer.compute_ic("momentum_20d")
print(analyzer.ic_summary("momentum_20d"))
```
""")
