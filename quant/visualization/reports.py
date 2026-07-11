"""
报告生成器
HTML / Markdown 格式的绩效报告输出
"""

from datetime import date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class ReportGenerator:
    """报告生成器

    生成 HTML 格式的量化分析报告
    """

    @staticmethod
    def generate_html_report(
        metrics: dict,
        equity_df: pd.DataFrame,
        trade_df: pd.DataFrame,
        output_path: str = "report.html",
        title: str = "量化回测报告",
    ) -> str:
        """生成 HTML 绩效报告"""
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Microsoft YaHei', sans-serif; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 40px; border-radius: 12px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
        .header p {{ opacity: 0.9; }}
        .card {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
        .card h2 {{ font-size: 20px; margin-bottom: 16px; color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 8px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }}
        .metric {{ background: #f8f9ff; padding: 16px; border-radius: 8px; text-align: center; }}
        .metric .value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
        .metric .label {{ font-size: 13px; color: #888; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9ff; font-weight: 600; color: #555; }}
        .positive {{ color: #52c41a; }}
        .negative {{ color: #ff4d4f; }}
        .footer {{ text-align: center; padding: 20px; color: #999; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {title}</h1>
            <p>生成日期: {date.today()} | 羊量量化平台</p>
        </div>

        <div class="card">
            <h2>核心指标</h2>
            <div class="metric-grid">
"""
        # 指标卡片
        for key, value in metrics.items():
            if isinstance(value, str) and "%" in value:
                cls = "positive" if value.startswith("+") or value.startswith("0%") is False else "negative"
                html += f"""
                <div class="metric">
                    <div class="value {cls}">{value}</div>
                    <div class="label">{key}</div>
                </div>"""

        html += """
            </div>
        </div>

        <div class="card">
            <h2>交易统计</h2>
            <table>
                <tr><th>指标</th><th>数值</th></tr>
"""
        if not trade_df.empty:
            buy_count = len(trade_df[trade_df["direction"] == "BUY"])
            sell_count = len(trade_df[trade_df["direction"] == "SELL"])
            total_cost = trade_df["total_cost"].sum() if "total_cost" in trade_df.columns else 0
            html += f"""
                <tr><td>买入笔数</td><td>{buy_count}</td></tr>
                <tr><td>卖出笔数</td><td>{sell_count}</td></tr>
                <tr><td>总交易成本</td><td>¥{total_cost:,.2f}</td></tr>
            """

        html += f"""
            </table>
        </div>

        <div class="footer">
            <p>🤖 由 羊量量化平台 (Yang Quant) 生成 | v0.1.0</p>
        </div>
    </div>
</body>
</html>"""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path

    @staticmethod
    def generate_markdown_report(
        metrics: dict,
        equity_df: pd.DataFrame,
        trade_df: pd.DataFrame,
        output_path: str = "report.md",
    ) -> str:
        """生成 Markdown 报告"""
        md = f"""# 📊 量化回测报告

> 生成日期: {date.today()} | 羊量量化平台

---

## 核心指标

| 指标 | 数值 |
|------|------|
"""
        for key, value in metrics.items():
            md += f"| {key} | {value} |\n"

        if not trade_df.empty:
            md += f"""
---

## 交易统计

| 指标 | 数值 |
|------|------|
| 买入笔数 | {len(trade_df[trade_df['direction'] == 'BUY'])} |
| 卖出笔数 | {len(trade_df[trade_df['direction'] == 'SELL'])} |
| 总交易 | {len(trade_df)} |
"""

        md += """

---

*🤖 由 羊量量化平台 生成*
"""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        return output_path
