"""
止盈止损策略回测
模拟：在随机日期买入股票，跟踪后续N天的盈亏，
测试当前 SellAdvisor 的止损/止盈规则是否合理。

结论会输出到控制台，方便直观看到问题。
"""

import sys, os, time, random, json, io
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# Fix Windows GBK encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─── 工具函数 ────────────────────────────
def get_kline(symbol, days=120):
    """从腾讯API获取日K线"""
    import requests
    import urllib3
    urllib3.disable_warnings()

    code = f"sh{symbol}" if symbol.startswith(("6", "9")) else f"sz{symbol}"
    url = (
        f"https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
        f"?param={code},day,,,{days},qfq"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.qq.com/",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        data = raw.get("data", {}).get(code, {}).get("qfqday", []) or raw.get("data", {}).get(code, {}).get("day", [])
        if not data:
            return None
        rows = []
        for d in data:
            rows.append({
                "date": d[0],
                "open": float(d[1]), "close": float(d[2]),
                "high": float(d[3]), "low": float(d[4]),
                "volume": float(d[5]),
            })
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        return None


def calc_atr(df, period=14):
    """计算ATR"""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr, index=df.index).rolling(period).mean()
    return atr


# ─── 当前策略规则 ─────────────────────────
def current_sell_rules(style="balanced"):
    """返回当前 SellAdvisor 的规则"""
    if style == "balanced":
        return {
            "name": "当前-平衡型",
            "stop_loss_pct": -0.05,
            "take_profit_1_pct": 0.06,    # 半仓止盈1
            "take_profit_2_pct": 0.12,    # 半仓止盈2
            "trailing_pct": 0.06,
            "max_hold_days": 10,
            "tp1_ratio": 0.5,  # 止盈1卖一半
        }
    else:
        return {
            "name": "当前-激进型",
            "stop_loss_pct": -0.07,
            "take_profit_1_pct": 0.09,
            "take_profit_2_pct": 0.18,
            "trailing_pct": 0.09,
            "max_hold_days": 7,
            "tp1_ratio": 0.5,
        }


# ─── 模拟一次持仓 ─────────────────────────
def simulate_hold(df, buy_idx, rules, use_atr=True):
    """
    在 buy_idx 这天以收盘价买入，跟踪后续走势。
    返回: {outcome, pnl_pct, hold_days, exit_reason, exit_price, max_profit, max_loss}
    """
    buy_price = df["close"].iloc[buy_idx]
    buy_date = df["date"].iloc[buy_idx]

    # 计算ATR
    atr_series = calc_atr(df)

    # 初始止损止盈价
    if use_atr and buy_idx >= 14:
        atr_val = atr_series.iloc[buy_idx]
        atr_pct = atr_val / buy_price if buy_price > 0 else 0.02
        stop_loss_price = buy_price * (1 - 2.0 * atr_pct)     # 2xATR止损
        tp1_price = buy_price * (1 + 1.5 * atr_pct)           # 1.5xATR止盈1
        tp2_price = buy_price * (1 + 3.0 * atr_pct)           # 3.0xATR止盈2
        trailing_trigger = buy_price * (1 + 2.0 * atr_pct)    # 移动止损触发
    else:
        stop_loss_price = buy_price * (1 + rules["stop_loss_pct"])
        tp1_price = buy_price * (1 + rules["take_profit_1_pct"])
        tp2_price = buy_price * (1 + rules["take_profit_2_pct"])
        trailing_trigger = buy_price * (1 + rules["trailing_pct"])

    max_hold = rules["max_hold_days"]
    tp1_ratio = rules["tp1_ratio"]

    # 跟踪
    max_price = buy_price
    min_price = buy_price
    highest_since_buy = buy_price
    trailing_stop_active = False
    trailing_stop_price = 0

    for i in range(buy_idx + 1, min(buy_idx + max_hold + 1, len(df))):
        row = df.iloc[i]
        high = row["high"]
        low = row["low"]
        close = row["close"]
        day = row["date"]
        hold_days = i - buy_idx

        max_price = max(max_price, high)
        min_price = min(min_price, low)

        # T+1 第一天不能卖，跳过
        if hold_days == 1:
            # 更新最高价但不触发卖出
            highest_since_buy = max(highest_since_buy, close)
            continue

        # 更新移动止损
        if not trailing_stop_active and close >= trailing_trigger:
            trailing_stop_active = True
            trailing_stop_price = buy_price  # 保本止损
        if trailing_stop_active:
            # 从最高点回撤6%(平衡)或9%(激进)
            highest_since_buy = max(highest_since_buy, close)
            trail_pct = rules["trailing_pct"]
            trailing_stop_price = max(
                trailing_stop_price,
                highest_since_buy * (1 - trail_pct)
            )

        # 检查止损（盘中最低价触及即触发）
        effective_stop = max(stop_loss_price, trailing_stop_price) if trailing_stop_active else stop_loss_price

        if low <= effective_stop:
            exit_price = effective_stop
            pnl_pct = (exit_price / buy_price - 1)
            return {
                "outcome": "止损",
                "pnl_pct": round(pnl_pct, 4),
                "hold_days": hold_days,
                "exit_reason": "移动止损" if trailing_stop_active else "初始止损",
                "exit_price": exit_price,
                "max_profit": round((max_price / buy_price - 1), 4),
                "max_loss": round((min_price / buy_price - 1), 4),
            }

        # 检查止盈1（卖一半）
        if high >= tp1_price and tp1_ratio > 0:
            # 止盈1触发：卖一半，剩余继续跟踪
            # 简化处理：如果后续达到tp2则按tp2算，否则按tp1
            # 这里只标记，继续循环
            pass

        # 检查止盈2
        if high >= tp2_price:
            # 计算综合收益：一半在tp1，一半在tp2
            avg_exit = tp1_price * tp1_ratio + tp2_price * (1 - tp1_ratio)
            pnl_pct = (avg_exit / buy_price - 1)
            return {
                "outcome": "止盈",
                "pnl_pct": round(pnl_pct, 4),
                "hold_days": hold_days,
                "exit_reason": "止盈触发",
                "exit_price": tp2_price,
                "max_profit": round((max_price / buy_price - 1), 4),
                "max_loss": round((min_price / buy_price - 1), 4),
            }

    # 持仓到期，以最后一天收盘价卖出
    last_idx = min(buy_idx + max_hold, len(df) - 1)
    last_row = df.iloc[last_idx]
    exit_price = last_row["close"]
    pnl_pct = (exit_price / buy_price - 1)
    return {
        "outcome": "到期平仓",
        "pnl_pct": round(pnl_pct, 4),
        "hold_days": last_idx - buy_idx,
        "exit_reason": f"持有{max_hold}天到期",
        "exit_price": exit_price,
        "max_profit": round((max_price / buy_price - 1), 4),
        "max_loss": round((min_price / buy_price - 1), 4),
    }


# ─── 改进策略规则 ─────────────────────────
def improved_sell_rules(style="balanced"):
    """改进后的规则（回测比较用）"""
    if style == "balanced":
        return {
            "name": "改进-平衡型",
            "stop_loss_pct": -0.06,         # 放宽到-6%
            "take_profit_1_pct": 0.05,      # 第一个止盈目标降低
            "take_profit_2_pct": 0.10,      # 第二个止盈目标
            "trailing_pct": 0.05,           # 移动止损从最高点回撤5%
            "max_hold_days": 10,
            "tp1_ratio": 0.5,
        }
    else:
        return {
            "name": "改进-激进型",
            "stop_loss_pct": -0.08,
            "take_profit_1_pct": 0.08,
            "take_profit_2_pct": 0.15,
            "trailing_pct": 0.07,
            "max_hold_days": 7,
            "tp1_ratio": 0.5,
        }


# ─── 主回测逻辑 ───────────────────────────
def run_backtest(stock_count=100, days_history=120):
    """主回测"""
    import akshare as ak

    print("=" * 70)
    print("  止盈止损策略回测")
    print("=" * 70)

    # 1. 获取股票池
    print("\n📊 获取股票池...")
    try:
        stock_df = ak.stock_info_a_code_name()
        # 过滤主板
        main_board = stock_df[stock_df["code"].str.match(r"^(60|00)\d{4}$")]
        # 排除ST
        main_board = main_board[~main_board["name"].str.contains("ST|退", na=False)]
        all_codes = main_board["code"].tolist()
        print(f"  主板股票: {len(all_codes)} 只")
    except Exception as e:
        print(f"  ❌ 获取股票池失败: {e}")
        return

    # 随机采样
    if len(all_codes) > stock_count:
        random.seed(42)
        samples = random.sample(all_codes, stock_count)
    else:
        samples = all_codes

    print(f"  采样: {len(samples)} 只")
    print(f"  每只下载 {days_history} 天K线...")

    # 2. 下载数据
    stock_data = {}
    success = 0
    for i, sym in enumerate(samples):
        df = get_kline(sym, days_history)
        if df is not None and len(df) >= 60:
            stock_data[sym] = df
            success += 1
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(samples)} (成功{success})")
        time.sleep(random.uniform(0.01, 0.05))

    print(f"  下载完成: {success}/{len(samples)} 只有效数据")

    if success < 10:
        print("  ❌ 有效数据太少，退出")
        return

    # 3. 对每只股票模拟多次买入
    all_results = {"当前-平衡型": [], "当前-激进型": [], "改进-平衡型": [], "改进-激进型": []}

    for sym, df in stock_data.items():
        n = len(df)
        # 在多个随机点位买入（每隔10天买一次）
        buy_points = list(range(30, n - 15, 10))
        if len(buy_points) > 10:
            buy_points = random.sample(buy_points, 10)  # 每只股票最多测10次

        for bp in buy_points:
            for rules_func, key in [
                (current_sell_rules, "当前-平衡型"),
                (current_sell_rules, "当前-激进型"),
                (improved_sell_rules, "改进-平衡型"),
                (improved_sell_rules, "改进-激进型"),
            ]:
                style = "balanced" if "平衡" in key else "aggressive"
                rules = rules_func(style)
                result = simulate_hold(df, bp, rules)
                result["symbol"] = sym
                result["buy_date"] = str(df["date"].iloc[bp])[:10]
                result["buy_price"] = round(df["close"].iloc[bp], 2)
                all_results[key].append(result)

    # 4. 汇总统计
    print("\n" + "=" * 70)
    print("  回测结果汇总")
    print("=" * 70)

    for strategy, results in all_results.items():
        if not results:
            continue

        n = len(results)
        pnls = [r["pnl_pct"] for r in results]

        # 分类统计
        outcomes = defaultdict(list)
        for r in results:
            outcomes[r["outcome"]].append(r["pnl_pct"])

        win_count = sum(1 for p in pnls if p > 0)
        loss_count = sum(1 for p in pnls if p < 0)

        avg_pnl = np.mean(pnls)
        median_pnl = np.median(pnls)
        total_pnl = sum(pnls)  # 累加收益

        # 计算盈亏比
        avg_win = np.mean([p for p in pnls if p > 0]) if win_count > 0 else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if loss_count > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        print(f"\n{'─' * 60}")
        print(f"  📌 {strategy}")
        print(f"{'─' * 60}")
        print(f"  总交易: {n} 笔")
        print(f"  胜率: {win_count/n*100:.1f}% ({win_count}赢/{loss_count}输)")
        print(f"  平均收益: {avg_pnl*100:+.2f}%")
        print(f"  中位数收益: {median_pnl*100:+.2f}%")
        print(f"  累加收益: {total_pnl*100:+.2f}%")
        print(f"  平均盈利: {avg_win*100:+.2f}% | 平均亏损: {avg_loss*100:+.2f}%")
        print(f"  盈亏比: {profit_factor:.2f}")

        for outcome, opnls in sorted(outcomes.items()):
            print(f"  └ {outcome}: {len(opnls)}笔 ({len(opnls)/n*100:.1f}%), "
                  f"平均{np.mean(opnls)*100:+.2f}%")

        # 最大回撤（累计）
        cumsum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumsum)
        dd = cumsum - peak
        max_dd = dd.min()
        print(f"  最大回撤(累计): {max_dd*100:.2f}%")

    # 5. 对比结论
    print("\n" + "=" * 70)
    print("  📊 对比总结")
    print("=" * 70)

    for style_prefix in ["当前", "改进"]:
        bal_key = f"{style_prefix}-平衡型"
        agg_key = f"{style_prefix}-激进型"
        if bal_key in all_results and agg_key in all_results:
            bal_avg = np.mean([r["pnl_pct"] for r in all_results[bal_key]]) * 100
            agg_avg = np.mean([r["pnl_pct"] for r in all_results[agg_key]]) * 100
            bal_win = sum(1 for r in all_results[bal_key] if r["pnl_pct"] > 0) / len(all_results[bal_key]) * 100
            agg_win = sum(1 for r in all_results[agg_key] if r["pnl_pct"] > 0) / len(all_results[agg_key]) * 100

            # 止损触发率
            bal_stop = sum(1 for r in all_results[bal_key] if r["outcome"] == "止损") / len(all_results[bal_key]) * 100
            agg_stop = sum(1 for r in all_results[agg_key] if r["outcome"] == "止损") / len(all_results[agg_key]) * 100

            print(f"\n  {style_prefix}策略:")
            print(f"    平衡型: 均收益{bal_avg:+.2f}% | 胜率{bal_win:.1f}% | 止损率{bal_stop:.1f}%")
            print(f"    激进型: 均收益{agg_avg:+.2f}% | 胜率{agg_win:.1f}% | 止损率{agg_stop:.1f}%")

    return all_results


if __name__ == "__main__":
    run_backtest(stock_count=80)
