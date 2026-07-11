"""
数据管理页面 — 股票列表、数据下载、数据预览
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from datetime import date, timedelta

st.set_page_config(page_title="数据管理", page_icon="📊", layout="wide")

st.title("📊 数据管理")

# ─── 股票列表 ─────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 股票列表", "⬇️ 数据下载", "📁 数据预览"])

with tab1:
    st.subheader("A股股票列表")

    col1, col2 = st.columns(2)
    with col1:
        market = st.selectbox("市场", ["全部", "沪市(60xxxx)", "深市(00xxxx/30xxxx)"])
    with col2:
        search = st.text_input("搜索代码/名称", placeholder="如: 000001 或 平安")

    if st.button("加载股票列表", type="primary"):
        with st.spinner("正在加载..."):
            try:
                from quant.data.symbols import SymbolManager

                sm = SymbolManager()
                df = sm.load_stock_list()

                if market == "沪市(60xxxx)":
                    df = df[df["symbol"].str.startswith("6")]
                elif market == "深市(00xxxx/30xxxx)":
                    df = df[~df["symbol"].str.startswith("6")]

                if search:
                    df = df[
                        df["symbol"].str.contains(search) |
                        df["name"].str.contains(search)
                    ]

                st.dataframe(df, use_container_width=True)
                st.success(f"共 {len(df)} 只股票")
            except ImportError:
                st.warning("请先安装 akshare: pip install akshare")
            except Exception as e:
                st.error(f"加载失败: {e}")

with tab2:
    st.subheader("下载行情数据")

    col1, col2, col3 = st.columns(3)
    with col1:
        symbols_input = st.text_input(
            "股票代码", placeholder="000001,600519,000002",
            help="多个代码用逗号分隔"
        )
    with col2:
        start_date = st.date_input("起始日期", value=date.today() - timedelta(days=365))
    with col3:
        end_date = st.date_input("结束日期", value=date.today())

    adjust = st.radio("复权方式", ["qfq(前复权)", "hfq(后复权)", "不复权"], horizontal=True)

    if st.button("开始下载", type="primary"):
        if not symbols_input:
            st.warning("请输入股票代码")
        else:
            symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

            with st.spinner(f"正在下载 {len(symbols)} 只股票数据..."):
                try:
                    from quant.data.sources import AkshareSource
                    from quant.data.storage import DataStorage

                    source = AkshareSource()
                    adjust_map = {
                        "qfq(前复权)": "qfq",
                        "hfq(后复权)": "hfq",
                        "不复权": "",
                    }

                    df = source.get_daily(
                        symbols=symbols,
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust_map[adjust],
                    )

                    if not df.empty:
                        storage = DataStorage()
                        count = storage.save_daily(df)

                        st.success(f"✅ 下载完成! {len(df)} 条数据, 保存了 {count} 个文件")
                        st.dataframe(df.head(100), use_container_width=True)

                        # 简单统计
                        st.subheader("数据概览")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("股票数", df["symbol"].nunique())
                        with col2:
                            st.metric("交易日", df["date"].nunique())
                        with col3:
                            st.metric("数据条数", len(df))
                    else:
                        st.warning("未获取到数据，请检查股票代码或日期范围")
                except Exception as e:
                    st.error(f"下载失败: {e}")

with tab3:
    st.subheader("本地数据预览")

    data_dir = st.text_input("数据目录", value="./data/daily")

    if st.button("扫描本地数据"):
        data_path = Path(data_dir)
        if not data_path.exists():
            st.warning(f"目录不存在: {data_dir}")
        else:
            parquet_files = list(data_path.rglob("*.parquet"))
            if not parquet_files:
                st.info("未找到数据文件")
            else:
                # 统计
                symbols_set = set()
                years_set = set()
                for f in parquet_files:
                    parts = f.parts
                    if len(parts) >= 4:
                        symbols_set.add(parts[-3])
                        years_set.add(parts[-2])

                st.success(f"找到 {len(parquet_files)} 个文件")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("覆盖股票", len(symbols_set))
                with col2:
                    st.metric("覆盖年份", ", ".join(sorted(years_set)))

                # 预览最新文件
                latest_file = max(parquet_files, key=lambda f: f.stat().st_mtime)
                try:
                    df = pd.read_parquet(latest_file)
                    st.subheader(f"最新文件预览: {latest_file}")
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"读取失败: {e}")
