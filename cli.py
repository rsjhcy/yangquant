#!/usr/bin/env python3
"""
羊量量化平台 — CLI 入口

用法:
    python cli.py data download --symbols 000001,600519 --start 2024-01-01
    python cli.py backtest run --strategy ma_cross --symbols 000001,600519
    python cli.py factor compute --symbols 000001,600519 --factor momentum
    python cli.py dashboard
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import click

# 确保项目路径在 sys.path
sys.path.insert(0, str(Path(__file__).parent))

from quant.config import config
from quant.utils.logger import setup_logger


@click.group()
@click.version_option(version="0.1.0", prog_name="羊量量化平台")
@click.pass_context
def cli(ctx):
    """🐑 羊量量化平台 — A股个人量化交易系统

    数据采集 → 因子研究 → 策略回测 → 实盘执行
    """
    setup_logger()


# ═════════════════════════════════════════════════
# 数据命令
# ═════════════════════════════════════════════════

@cli.group()
def data():
    """📊 数据管理 — 下载、查询、更新行情数据"""
    pass


@data.command("download")
@click.option("--symbols", "-s", required=True,
              help="股票代码，多个用逗号分隔，如: 000001,600519")
@click.option("--start", required=True, help="起始日期 YYYY-MM-DD")
@click.option("--end", help="结束日期 YYYY-MM-DD (默认今天)")
@click.option("--adjust", default="qfq",
              type=click.Choice(["qfq", "hfq", ""]),
              help="复权方式 (默认前复权)")
@click.option("--output", "-o", default="./data",
              help="数据存储目录")
def download(symbols, start, end, adjust, output):
    """下载A股日线数据"""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    symbols_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbols_list:
        click.echo("❌ 请提供至少一个股票代码", err=True)
        return

    click.echo(f"📥 下载 {len(symbols_list)} 只股票数据...")
    click.echo(f"   日期: {start_date} ~ {end_date}")
    click.echo(f"   复权: {adjust or '不复权'}")

    from quant.data.sources import AkshareSource
    from quant.data.storage import DataStorage

    source = AkshareSource()
    storage = DataStorage(root_dir=output)

    try:
        df = source.get_daily(symbols_list, start_date, end_date, adjust=adjust)

        if df.empty:
            click.echo("⚠ 未获取到数据，请检查代码或网络")
            return

        count = storage.save_daily(df)
        click.echo(f"✅ 成功! {len(df)} 条数据 → {count} 个文件")
        click.echo(f"   股票数: {df['symbol'].nunique()}")
        click.echo(f"   交易日: {df['date'].nunique()}")
        click.echo(f"   存储路径: {Path(output).absolute()}")

        # 数据摘要
        for sym in symbols_list[:5]:
            sym_data = df[df["symbol"] == sym]
            if not sym_data.empty:
                click.echo(f"   {sym}: {sym_data['close'].iloc[0]:.2f} → {sym_data['close'].iloc[-1]:.2f}")

    except Exception as e:
        click.echo(f"❌ 下载失败: {e}", err=True)


@data.command("list")
@click.option("--market", default="all",
              type=click.Choice(["all", "sh", "sz"]),
              help="过滤市场")
def list_stocks(market):
    """列出A股股票"""
    click.echo(f"📋 加载A股列表...")

    from quant.data.symbols import SymbolManager

    sm = SymbolManager()
    df = sm.load_stock_list()

    if market == "sh":
        df = df[df["symbol"].str.startswith("6")]
    elif market == "sz":
        df = df[~df["symbol"].str.startswith("6")]

    click.echo(f"   共 {len(df)} 只股票")
    click.echo(df[["symbol", "name", "industry"]].head(50).to_string(index=False))


@data.command("watch")
@click.option("--add", "-a", multiple=True, help="添加股票到关注列表")
@click.option("--set", "-s", help="设置关注列表 (逗号分隔, 覆盖原有)")
@click.option("--show", "show_list", is_flag=True, help="查看当前关注列表")
def watch(add, set, show_list):
    """管理股票关注列表"""
    from quant.data.updater import DataUpdater

    updater = DataUpdater()

    if set:
        symbols = [s.strip() for s in set.split(",") if s.strip()]
        updater.set_watchlist(symbols)
        click.echo(f"关注列表已设置: {len(symbols)} 只股票")
        for sym in symbols:
            click.echo(f"  - {sym}")

    if add:
        updater.add_to_watchlist(list(add))
        click.echo(f"已添加 {len(add)} 只股票")

    show_list = show_list or (not set and not add)
    if show_list:
        watchlist = updater.load_watchlist()
        if watchlist:
            click.echo(f"当前关注列表 ({len(watchlist)} 只):")
            for sym in watchlist:
                click.echo(f"  - {sym}")
        else:
            click.echo("关注列表为空。请使用 --set 或 --add 添加:")
            click.echo("  python cli.py data watch --set 000001,600519,000858")


@data.command("update")
@click.option("--symbols", "-s", default=None, help="要更新的股票 (默认=关注列表)")
@click.option("--daily", is_flag=True, help="启动每日自动更新 (收盘后)")
def update_data(symbols, daily):
    """增量更新数据 (只下载本地缺失的部分)"""
    from quant.data.updater import DataUpdater

    updater = DataUpdater()

    if daily:
        click.echo("启动每日自动更新 (收盘后 15:30)...")
        click.echo("按 Ctrl+C 停止\n")
        syms = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
        updater.run_daily(syms)
        return

    syms = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
    result = updater.update_all(syms)

    click.echo(f"\n更新结果: {result['success']} 成功, {result['failed']} 失败, {result['new_rows']} 条新数据")

    # 显示状态
    click.echo("\n数据覆盖状态:")
    status_df = updater.status()
    if not status_df.empty:
        click.echo(status_df.to_string(index=False))


@data.command("calendar")
@click.option("--year", default=None, type=int, help="年份")
def show_calendar(year):
    """显示交易日历"""
    from quant.data.calendar import cal

    cal.load()
    y = year or date.today().year
    start = date(y, 1, 1)
    end = date(y, 12, 31)
    days = cal.get_trading_days(start, end)

    click.echo(f"📅 {y}年 共有 {len(days)} 个交易日")
    click.echo(f"   起止: {days[0]} ~ {days[-1]}")


# ═════════════════════════════════════════════════
# 回测命令
# ═════════════════════════════════════════════════

@cli.group()
def backtest():
    """📈 策略回测 — 运行和评估交易策略"""
    pass


@backtest.command("run")
@click.option("--strategy", "-s", default="ma_cross",
              type=click.Choice(["ma_cross", "mean_reversion", "alpha_momentum"]),
              help="策略名称")
@click.option("--symbols", required=True, help="股票代码(逗号分隔)")
@click.option("--start", required=True, help="起始日期 YYYY-MM-DD")
@click.option("--end", help="结束日期 YYYY-MM-DD")
@click.option("--capital", default=1_000_000, help="初始资金")
@click.option("--fast", default=5, help="快线周期 (ma_cross)")
@click.option("--slow", default=20, help="慢线周期 (ma_cross)")
@click.option("--top-n", default=10, help="持仓数量 (alpha_momentum)")
@click.option("--report", "-r", is_flag=True, help="生成HTML报告")
def run_backtest(strategy, symbols, start, end, capital, fast, slow, top_n, report):
    """运行策略回测"""
    from quant.backtest import BacktestEngine, PerformanceAnalytics
    from quant.data.sources import AkshareSource
    from quant.strategy.examples import (
        MACrossoverStrategy,
        MeanReversionStrategy,
        AlphaMomentumStrategy,
    )

    symbols_list = [s.strip() for s in symbols.split(",") if s.strip()]
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    click.echo(f"🚀 开始回测")
    click.echo(f"   策略: {strategy}")
    click.echo(f"   股票: {symbols_list}")
    click.echo(f"   日期: {start_date} ~ {end_date}")
    click.echo(f"   资金: ¥{capital:,}")

    # 获取数据
    click.echo("📥 加载行情数据...")
    source = AkshareSource()
    df = source.get_daily(symbols_list, start_date, end_date)

    if df.empty:
        click.echo("❌ 无行情数据")
        return

    click.echo(f"   获取 {len(df)} 条数据")

    # 创建引擎
    engine = BacktestEngine(initial_cash=capital)
    engine.set_data(df)

    strategy_map = {
        "ma_cross": MACrossoverStrategy,
        "mean_reversion": MeanReversionStrategy,
        "alpha_momentum": AlphaMomentumStrategy,
    }

    strat = strategy_map[strategy](**({"fast": fast, "slow": slow} if strategy == "ma_cross" else {}))
    engine.set_strategy(strat)

    # 运行
    click.echo("⚙ 运行回测...")
    result = engine.run(progress=True)

    # 输出报告
    click.echo("\n" + "=" * 50)
    click.echo("📊 回测结果")
    click.echo("=" * 50)

    r = result.describe()
    for k, v in r.items():
        if isinstance(v, float):
            click.echo(f"  {k}: {v:.4f}")
        else:
            click.echo(f"  {k}: {v}")

    # HTML 报告
    if report:
        from quant.visualization.reports import ReportGenerator

        path = ReportGenerator.generate_html_report(
            r, result.equity_curve, result.trade_log
        )
        click.echo(f"\n📄 报告已保存: {path}")


# ═════════════════════════════════════════════════
# 因子命令
# ═════════════════════════════════════════════════

@cli.group()
def factor():
    """🔬 因子研究 — 计算和分析量化因子"""
    pass


@factor.command("compute")
@click.option("--symbols", "-s", required=True, help="股票代码(逗号分隔)")
@click.option("--factor-name", "-f", default="momentum",
              type=click.Choice(["momentum", "rsi", "volatility", "ma_deviation", "turnover"]),
              help="因子名称")
@click.option("--start", required=True, help="起始日期 YYYY-MM-DD")
@click.option("--end", help="结束日期 YYYY-MM-DD")
@click.option("--period", default=20, help="因子计算周期")
def compute_factor(symbols, factor_name, start, end, period):
    """计算因子值"""
    from quant.data.sources import AkshareSource
    from quant.factors.technical import (
        MomentumFactor,
        RSIFactor,
        VolatilityFactor,
        MADeviationFactor,
        TurnoverFactor,
    )

    symbols_list = [s.strip() for s in symbols.split(",") if s.strip()]
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    click.echo(f"🔬 计算因子: {factor_name}")
    click.echo(f"   股票: {len(symbols_list)} 只")
    click.echo(f"   日期: {start_date} ~ {end_date}")

    source = AkshareSource()
    df = source.get_daily(symbols_list, start_date, end_date)

    factor_map = {
        "momentum": MomentumFactor(period=period),
        "rsi": RSIFactor(period=period),
        "volatility": VolatilityFactor(period=period),
        "ma_deviation": MADeviationFactor(ma_period=period),
        "turnover": TurnoverFactor(short=5, long=20),
    }

    factor_obj = factor_map[factor_name]
    result = factor_obj.compute(df)

    click.echo(f"\n📊 {result.name} 因子值:")
    for symbol, val in result.values.sort_values(ascending=False).items():
        if not (val != val):  # NaN check
            click.echo(f"  {symbol}: {val:.4f}")
        else:
            click.echo(f"  {symbol}: NaN")


@factor.command("analyze")
@click.option("--symbols", "-s", required=True, help="股票代码")
@click.option("--factor-name", "-f", default="momentum",
              type=click.Choice(["momentum", "rsi", "volatility"]))
@click.option("--start", required=True, help="起始日期 YYYY-MM-DD")
@click.option("--end", help="结束日期 YYYY-MM-DD")
def analyze_factor(symbols, factor_name, start, end):
    """分析因子 (IC分析、分层回测)"""
    click.echo("🔬 因子分析 (需要多日数据)")
    click.echo("   此功能在完整 Jupyter Notebook 中效果更好")
    click.echo("   参考: notebooks/01_因子探索.ipynb")


# ═════════════════════════════════════════════════
# Dashboard
# ═════════════════════════════════════════════════

@cli.command()
@click.option("--port", default=8501, help="端口号")
def dashboard(port):
    """🌐 启动 Streamlit Dashboard"""
    import subprocess

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    click.echo(f"🌐 启动 Dashboard: http://localhost:{port}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
    ])


# ═════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════

if __name__ == "__main__":
    cli()
