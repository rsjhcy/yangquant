"""
Streamlit 可复用图表组件
"""

import plotly.graph_objects as go
import streamlit as st


def metrics_row(metrics: dict, columns: int = 4):
    """关键指标行"""
    cols = st.columns(columns)
    items = list(metrics.items())
    for i, (label, value) in enumerate(items):
        with cols[i % columns]:
            st.metric(label=label, value=value)


def plot_equity_curve(fig: go.Figure, height: int = 400):
    """净值曲线"""
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


def plot_drawdown(fig: go.Figure, height: int = 250):
    """回撤图"""
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


def plot_heatmap(fig: go.Figure, height: int = 400):
    """热力图"""
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


def show_dataframe(df, title: str = "", height: int = 300):
    """展示 DataFrame"""
    if title:
        st.subheader(title)
    st.dataframe(df, use_container_width=True, height=height)
