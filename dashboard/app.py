"""
羊量量化平台 — Streamlit Dashboard 主入口

启动:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="羊量量化平台",
    page_icon="🐑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🐑 羊量量化平台")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 导航")
st.sidebar.markdown("- 📊 数据管理")
st.sidebar.markdown("- 🔬 因子研究")
st.sidebar.markdown("- 📈 策略回测")
st.sidebar.markdown("- 💹 实盘监控")
st.sidebar.markdown("---")
st.sidebar.info(
    "**羊量量化 v0.1.0**\n\n"
    "A股全流程量化平台\n\n"
    "数据采集 → 因子研究 → 策略回测 → 实盘执行"
)

# 首页
st.title("🐑 羊量量化平台")
st.markdown("### 你的个人A股量化工作站")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📦 数据下载", "就绪", "akshare")

with col2:
    st.metric("🔬 内置因子", "8+", "技术/基本面")

with col3:
    st.metric("📈 示例策略", "3", "可扩展")

with col4:
    st.metric("⚡ 回测引擎", "事件驱动", "A股约束")

st.markdown("---")

# 快速开始
st.subheader("🚀 快速开始")

code_tab1, code_tab2, code_tab3 = st.tabs(["1. 下载数据", "2. 运行回测", "3. 因子研究"])

with code_tab1:
    st.code("""
# 通过 CLI 下载数据
python cli.py data download --symbols 000001,600519 --start 2024-01-01 --end 2024-12-31

# 或在 Python 中
from quant.data.sources import AkshareSource
from quant.data.storage import DataStorage
from datetime import date

source = AkshareSource()
storage = DataStorage()

df = source.get_daily(['000001'], date(2024,1,1), date(2024,12,31))
storage.save_daily(df)
print(f"保存了 {len(df)} 条数据")
""", language="python")

with code_tab2:
    st.code("""
from quant.backtest import BacktestEngine
from quant.strategy.examples import MACrossoverStrategy
from quant.data.sources import AkshareSource

# 获取数据
source = AkshareSource()
df = source.get_daily(['000001', '600519'], date(2023,1,1), date(2023,12,31))

# 创建回测
engine = BacktestEngine(initial_cash=1_000_000)
engine.set_data(df)
engine.set_strategy(MACrossoverStrategy(fast=5, slow=20))

# 运行
result = engine.run()

# 查看结果
print(f"总收益: {result.total_return:.2%}")
print(result.describe())
""", language="python")

with code_tab3:
    st.code("""
from quant.factors import MomentumFactor, RSIFactor, FactorAnalyzer
from quant.data.sources import AkshareSource

source = AkshareSource()
df = source.get_daily(['000001', '000002', '600519'], date(2024,1,1), date(2024,6,1))

# 计算因子
mom = MomentumFactor(period=20)
result = mom.compute(df)
print(result.values)
""", language="python")

st.markdown("---")
st.caption("🤖 羊量量化平台 | 用心构建你的量化未来")
